# HYBRID MODEL TRAINING SCRIPT
#%% IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import audioread
import warnings

warnings.filterwarnings("ignore")


#%% CONFIGURACIÓN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir("../")
data_dir = os.getcwd() + "/dataset/train"

#%% FUNCIONES

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
    

audio_files, vocal_files = load_data(data_dir)
audio_files, vocal_files = split_songs(audio_files, vocal_files)
dataset = AudioDataset(audio_files, vocal_files)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

#%% HYBRID MODEL: CONV + RNN
class VocalSeparationHybrid(nn.Module):
    def __init__(self, n_mels=128, conv_filters=64, kernel_size=3, hidden_size=256, num_layers=2, output_size=128, bidirectional=True):
        super(VocalSeparationHybrid, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, conv_filters, kernel_size=(kernel_size, kernel_size), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_filters)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=(kernel_size, kernel_size), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_filters * 2)
        
        # Update LSTM input_size dynamically based on conv_filters
        self.lstm_input_size = n_mels * (conv_filters * 2)  # Channels * n_mels
        
        # Recurrent layers
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Fully connected layer to map back to the original spectrogram dimensions
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        # x shape: [batch, channels, n_mels, time]
        x = x.unsqueeze(1)  # Add channel dimension for 2D convolutions
        
        # Convolutional layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        
        # Reshape for RNN: [batch, time, features]
        batch, channels, n_mels, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, -1)  # Combine channels and n_mels
        
        # Check dimensions for debugging
        assert x.size(-1) == self.lstm_input_size, f"Expected LSTM input size {self.lstm_input_size}, got {x.size(-1)}"
        
        # Recurrent layers
        lstm_out, _ = self.lstm(x)
        
        # Fully connected layer
        output = self.fc(lstm_out)
        return output.transpose(1, 2)  # Reshape back to [batch, output_size, time]

#%% ENTRENAMIENTO
model = VocalSeparationHybrid().to(device)
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.L1Loss()

epochs = 100
losses = []
for epoch in range(epochs):
    loss = train(model, dataloader, optimizer, criterion, device)
    losses.append(loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

#%% GRAFICAR Y GUARDAR
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig(f'plots/hybrid_loss_{epochs}epochs_{lr}lr_L1.png')
torch.save(model.state_dict(), f'models/hybrid_model_{epochs}epochs_{lr}lr_L1.pth')