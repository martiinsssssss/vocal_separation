# CROSS-DOMAIN TRANSFER LEARNING MODEL TRAINING SCRIPT

#%% IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
import warnings
import audioread

warnings.filterwarnings("ignore")

os.chdir("../")
data_dir = data_dir = os.getcwd() + "/dataset/train"
print(data_dir)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% FUNCTIONS

# Function to load the data 
def load_data(data_dir):
    """
    This function loads the audio files and vocal files from the given data directory.

    Args:
    data_dir (str): Path to the directory containing the audio files and vocal files.
    """

    audio_files = []
    vocal_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".stem_vocals.mp3"):
                vocal_file = os.path.join(root, file)
                audio_file = vocal_file.replace(".stem_vocals.mp3", ".stem.mp3")
                if os.path.exists(audio_file):
                    audio_files.append(audio_file)
                    vocal_files.append(vocal_file)
    return audio_files, vocal_files

def split_songs(audio_files, vocal_files, sr=22050, cut=20):
    """
    This function splits the given audio file into 20 segments.

    Args:
    audio_file (str): Path to the audio file to split.

    Returns:
    list: List of 20s audio segments.
    """
    all_audios = []
    all_vocals = []
    for i, (audio_file, vocal_file) in enumerate(zip(audio_files, vocal_files)):

        mix, _ = librosa.load(audioread.audio_open(audio_file), sr=sr)
        vocals, _ = librosa.load(audioread.audio_open(vocal_file), sr=sr)

        # split the audio into 20s segments
        for j in range(0, len(mix), sr*cut):
            if j + sr*cut < len(mix):
                all_audios.append(mix[j:j+sr*cut])
                all_vocals.append(vocals[j:j+sr*cut])

    return all_audios, all_vocals

def collate_fn(batch):
    mix_db_list, vocals_db_list = zip(*batch)
    max_len = max([x.shape[-1] for x in mix_db_list])
    mix_db_list = [torch.cat([x, torch.zeros((x.shape[0], max_len - x.shape[1]))], dim=-1) for x in mix_db_list]
    vocals_db_list = [torch.cat([x, torch.zeros((x.shape[0], max_len - x.shape[1]))], dim=-1) for x in vocals_db_list]
    return torch.stack(mix_db_list, 0), torch.stack(vocals_db_list, 0)

def train(model, dataloader, optimizer, criterion, device):
    """
    This function trains the given model on the given dataloader for one epoch.

    Args:
    model (nn.Module): Model to train.
    dataloader (DataLoader): DataLoader containing the training data.
    optimizer (Optimizer): Optimizer to use for training.
    criterion (Loss): Loss function to use for training.
    device (str): Device to use for training (cpu or cuda).

    Returns:
    float: Average loss for this epoch.
    """

    model.train()
    running_loss = 0.0
    for mix_db, vocals_db in tqdm(dataloader, desc="Training"):
        mix_db, vocals_db = mix_db.to(device), vocals_db.to(device)

        optimizer.zero_grad()

        # Forward pass: Get model predictions
        outputs = model(mix_db)  # Add channel dimension

        # Pad vocals_db to match outputs
        if outputs.shape[-1] > vocals_db.shape[-1]:
            padding = outputs.shape[-1] - vocals_db.shape[-1]
            vocals_db = F.pad(vocals_db, (0, padding), mode='constant', value=0)
        elif outputs.shape[-1] < vocals_db.shape[-1]:
            padding = vocals_db.shape[-1] - outputs.shape[-1]
            outputs = F.pad(outputs, (0, padding), mode='constant', value=0)

        # Compute the loss
        loss = criterion(outputs, vocals_db)  # Add channel dimension to vocals
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        #scheduler.step()
        #scheduler.step(loss)

        running_loss += loss.item()
    
    return running_loss / len(dataloader)


#%% DATASET CLASS and DATALOADER
class AudioDataset(Dataset):
    def __init__(self, audio_files, vocal_files, sr=22050, n_mels=128, max_len=30000, cut=30):
        self.audio_files = audio_files
        self.vocal_files = vocal_files
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.cut = cut

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load audio and vocals
        mix = self.audio_files[idx]
        vocals = self.vocal_files[idx]

        # Compute Mel spectrograms
        mix_mel = librosa.feature.melspectrogram(y=mix, sr=self.sr, n_mels=self.n_mels)
        vocals_mel = librosa.feature.melspectrogram(y=vocals, sr=self.sr, n_mels=self.n_mels)

        # Convert Mel spectrograms to dB scale
        mix_db = librosa.power_to_db(mix_mel, ref=np.max)
        vocals_db = librosa.power_to_db(vocals_mel, ref=np.max)

        # Normalize spectrograms
        min_db, max_db = -80.0, 0.0  # Decibel range
        vocals_db = (vocals_db - min_db) / (max_db - min_db)

        # Dynamically adjust max_len
        max_len = min(self.max_len, mix_db.shape[-1])

        # Pad or truncate spectrograms
        if mix_db.shape[-1] < max_len:
            pad_width = max_len - mix_db.shape[-1]
            mix_db = np.pad(mix_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=1e-5)
            vocals_db = np.pad(vocals_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=1e-5)
        else:
            mix_db = mix_db[:, :max_len]
            vocals_db = vocals_db[:, :max_len]

        # Convert to torch tensors
        mix_db = torch.tensor(mix_db, dtype=torch.float32)
        vocals_db = torch.tensor(vocals_db, dtype=torch.float32)

        return mix_db, vocals_db


#%% LOAD DATA
audio_files, vocal_files = load_data(data_dir)
audio_files, vocal_files = split_songs(audio_files, vocal_files)
dataset = AudioDataset(audio_files, vocal_files)
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)

#%% PRETRAINED MODEL ADAPTATION
from torchvision.models import resnet34

class VocalSeparationPretrained(nn.Module):
    def __init__(self):
        super(VocalSeparationPretrained, self).__init__()
        # Load a pretrained ResNet
        pretrained_resnet = resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(pretrained_resnet.children())[:-2])  # Remove final layers
        
        # Custom decoder for vocal separation
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, H/32, W/32]
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # [B, 256, H/16, W/16]
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, H/16, W/16]
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # [B, 128, H/8, W/8]
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, H/8, W/8]
            nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),  # [B, 1, H, W]
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # Final vocal separation layer
            #nn.ReLU(),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert grayscale to 3 channels [B, 3, H, W]
        features = self.feature_extractor(x)  # Extract features
        output = self.decoder(features)  # Decode to vocal separation
        return output.squeeze(1)  # Remove channel dimension [B, H, W]

# Load a pretrained model (e.g., ResNet18)
model = VocalSeparationPretrained().to(device)

for param in model.feature_extractor.parameters():
    param.requires_grad = False

#%% TRAINING
lr = 1e-4
# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#criterion = nn.MSELoss()
criterion = nn.L1Loss()

# Train the model
epochs = 100
losses = []
for epoch in range(epochs):
    loss = train(model, dataloader, optimizer, criterion, device)
    losses.append(loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# Plot training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# save plot
plt.savefig(f'plots/pretrained_loss_{epochs}epochs_{lr}lr_L1.png')

# Save the model
torch.save(model.state_dict(), f'models/pretrained_model_{epochs}epochs_{lr}lr_L1.pth')