# CURRICULUM-LEARNING MODEL TRAINING SCRIPT

"""
This script is used to train the baseline CNN model for vocal separation using curriculum learning, where the model is trained on progressively 
harder examples.
"""

#%% IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#!pip install ffmpeg
import ffmpeg
import librosa
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pydub import AudioSegment
import soundfile as sf
from pydub import AudioSegment
from IPython.display import Audio
import audioread
import warnings

warnings.filterwarnings("ignore")

os.chdir("../")
data_dir = os.getcwd() + "/dataset/train"
print(data_dir)
test_dir = os.getcwd() + "/dataset/test"


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
    """
    This function collates a batch of samples into a single tensor.

    Args:
    batch (list): List of samples, where each sample is a tuple of the form (mix_db, vocals_db).

    Returns:
    mix_db_batch (Tensor): Tensor containing the mix spectrograms for the batch.
    vocals_db_batch (Tensor): Tensor containing the vocals spectrograms for the batch.
    """

    mix_db_list, vocals_db_list = zip(*batch)
    
    # Find the maximum length in this batch
    max_len = max([x.shape[-1] for x in mix_db_list])
    
    # Pad all samples to the maximum length in this batch
    mix_db_list = [torch.cat([x, torch.zeros((x.shape[0], max_len - x.shape[1]))], dim=-1) if x.shape[-1] < max_len else x for x in mix_db_list]
    vocals_db_list = [torch.cat([x, torch.zeros((x.shape[0], max_len - x.shape[1]))], dim=-1) if x.shape[-1] < max_len else x for x in vocals_db_list]
    
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
        
        # Compute the loss
        loss = criterion(outputs, vocals_db)  # Add channel dimension to vocals
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        #scheduler.step()
        #scheduler.step(loss)

        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """
    This function validates the model on the given dataloader.

    Args:
    model (nn.Module): The model to validate.
    dataloader (DataLoader): The DataLoader for the validation data.
    criterion (Loss): The loss function.
    device (str): The device to use (cpu or cuda).

    Returns:
    float: The average validation loss.
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for mix_db, vocals_db in tqdm(dataloader, desc="Validation"):
            mix_db, vocals_db = mix_db.to(device), vocals_db.to(device)

            # Forward pass
            outputs = model(mix_db)
            loss = criterion(outputs, vocals_db)
            running_loss += loss.item()

    return running_loss / len(dataloader)

#%% DATASET CLASS and DATALOADER

class AudioDataset(Dataset):
    def __init__(self, audio_files, vocal_files, sr=22050, n_mels=128, max_len=30000, cut = 30):
        self.audio_files = audio_files
        self.vocal_files = vocal_files
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len  # maximum length to pad or truncate the spectrogram to
        self.cut = cut # cut the audio to 30 secondsç        ç
        self.difficulties = self.compute_difficulties()


    def __len__(self):
        return len(self.audio_files)
    
    def compute_difficulties(self):
        difficulties = []
        for mix_file, vocal_file in zip(self.audio_files, self.vocal_files):
            #mix, _ = librosa.load(mix_file, sr=self.sr)
            #vocals, _ = librosa.load(vocal_file, sr=self.sr)
            vmr = np.sum(vocal_file ** 2) / np.sum(mix_file ** 2)
            difficulties.append(vmr)
        return difficulties
    
    def __getitem__(self, idx):
        # Load audio and vocals
        
        mix = audio_files[idx]
        vocals = vocal_files[idx]

        # Compute Mel spectrograms
        mix_mel = librosa.feature.melspectrogram(y=mix, sr=self.sr, n_mels=self.n_mels)
        vocals_mel = librosa.feature.melspectrogram(y=vocals, sr=self.sr, n_mels=self.n_mels)

        # Convert Mel spectrograms to dB scale
        mix_db = librosa.power_to_db(mix_mel, ref=np.max)
        vocals_db = librosa.power_to_db(vocals_mel, ref=np.max)

        ## Normalize spectrograms
        #mix_db = (mix_db - mix_db.mean()) / mix_db.std()
        #vocals_db = (vocals_db - vocals_db.mean()) / vocals_db.std()

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
    
    def get_audio(self, idx):
        mix, _ = librosa.load(audioread.audio_open(self.audio_files[idx]), sr=self.sr)
        vocals, _ = librosa.load(audioread.audio_open(self.vocal_files[idx]), sr=self.sr)

        if len(mix) > self.sr * self.cut:
            start = (len(mix) - self.sr * self.cut) // 2
            end = start + self.sr * self.cut
            mix = mix[start:end] 
            vocals = vocals[start:end]
        
        return mix, vocals
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

audio_files, vocal_files = load_data(data_dir)

# Create dataset and dataloader
audio_files, vocal_files = split_songs(audio_files, vocal_files)
dataset = AudioDataset(audio_files, vocal_files)

# Convert difficulties to a list if it's a numpy array
if isinstance(dataset.difficulties, np.ndarray):
    difficulties_list = dataset.difficulties.tolist()
else:
    difficulties_list = dataset.difficulties

# Zip the difficulties with the corresponding audio and vocal files
zipped_data = zip(difficulties_list, dataset.audio_files, dataset.vocal_files)

# Sort by difficulties (the first element of each tuple)
sorted_data = sorted(zipped_data, key=lambda x: x[0])  # Sort by difficulty

# Unzip the sorted data back into separate lists
dataset.difficulties, dataset.audio_files, dataset.vocal_files = zip(*sorted_data)

val_audio_files, val_vocal_files = load_data(test_dir)

# Create dataset and dataloader
val_audio_files, val_vocal_files = split_songs(val_audio_files, val_vocal_files)
val_dataset = AudioDataset(val_audio_files, val_vocal_files)
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

# print an example of a batch of data

mix_db_batch, vocals_db_batch = next(iter(val_dataloader))
print(mix_db_batch.shape, vocals_db_batch.shape)

#%% MODEL
class VocalSeparationBaseline(nn.Module):
    def __init__(self):
        super(VocalSeparationBaseline, self).__init__()
        # Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 128, 1292]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 128, 1292]
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # [B, 32, 64, 646]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 64, 646]
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),  # [B, 1, 64, 646]
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # [B, 1, 128, 1292]
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension [B, 1, 128, 1292]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)  # Remove channel dimension [B, 128, 1292]

#%% TRAINING

# Initialize the model, optimizer, and loss function
model = VocalSeparationBaseline().to(device)
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

criterion = nn.MSELoss()
#criterion = nn.L1Loss()

# Train the model
epochs = 100
train_losses = []
val_losses = []
for epoch in range(epochs):
    # Determine the proportion of data to use (e.g., start with 10%, end with 100%)
    curr_proportion = min(1.0, 0.1 + 0.9 * (epoch / epochs))  # Linearly increase from 10% to 100%
    subset_size = int(curr_proportion * len(dataset))
    
    # Subset the dataset
    train_subset = torch.utils.data.Subset(dataset, range(subset_size))
    dataloader = DataLoader(train_subset, batch_size=8, collate_fn=collate_fn, shuffle=True)

    train_loss = train(model, dataloader, optimizer, criterion, device)
    train_losses.append(train_loss)

    # Validation step
    val_loss = validate(model, val_dataloader, criterion, device)
    val_losses.append(val_loss)
    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(False)
plt.savefig(f'plots/curriculum_{epochs}epochs_{lr}lr_MSE.png', transparent = True)

# Save the model
torch.save(model.state_dict(), f'models/curriculum_model_{epochs}epochs_{lr}lr_MSE.pth')