# BASELINE CNN MODEL TRAINING SCRIPT

"""
This script is used to train the baseline model, which is a simple CNN model that takes in the audio spectrogram
as input and predicts the vocals. The model is trained on the preprocessed data from the raw database, which consists
of audio segments of 30s with the vocals extracted and saved as .mp3 files. The input audio files are converted to
.mp3 format and resampled to 22050 Hz. The script uses the Librosa library to compute the mel spectrogram of the audio
files, which is then used as input to the model. The model architecture consists of a series of convolutional layers
followed by max pooling layers, with a final dense layer to output the predicted vocals. The model is trained using
the Adam optimizer and the mean squared error loss function. The training data is split into training and validation
sets, with a batch size of 32 and 50 epochs of training. The model is evaluated on the validation set and saved to disk
if it achieves a lower validation loss than the previous best model.
"""

#%% IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
data_dir = data_dir = os.getcwd() + "/dataset/train"
print(data_dir)

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

#%% DATASET CLASS and DATALOADER

class AudioDataset(Dataset):
    def __init__(self, audio_files, vocal_files, sr=22050, n_mels=128, max_len=30000, cut = 30):
        self.audio_files = audio_files
        self.vocal_files = vocal_files
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len  # maximum length to pad or truncate the spectrogram to
        self.cut = cut # cut the audio to 30 seconds

    def __len__(self):
        return len(self.audio_files)

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
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)

# print an example of a batch of data

mix_db_batch, vocals_db_batch = next(iter(dataloader))
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

#criterion = nn.MSELoss()
criterion = nn.L1Loss()

# Train the model
epochs = 100
losses = []
for epoch in range(epochs):
    loss = train(model, dataloader, optimizer, criterion, device)
    losses.append(loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# Plot the training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# save plot
plt.savefig(f'plots/baseline_please_loss_{epochs}epochs_{lr}lr_L1.png')

# Save the model
torch.save(model.state_dict(), f'models/baseline_please_model_{epochs}epochs_{lr}lr_L1.pth')