# BASELINE RNN MODEL TRAINING SCRIPT
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
import warnings

warnings.filterwarnings("ignore")


#%% CONFIGURACIÓN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir("../")
data_dir = os.getcwd() + "/dataset/train"

#%% FUNCIONES

def load_data(data_dir):
    """
    Carga los archivos de audio y vocales desde el directorio especificado.
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

def collate_fn(batch):
    """
    Une un lote de muestras en un único tensor.
    """
    mix_db_list, vocals_db_list = zip(*batch)
    max_len = max([x.shape[-1] for x in mix_db_list])
    mix_db_list = [torch.cat([x, torch.zeros((x.shape[0], max_len - x.shape[1]))], dim=-1) if x.shape[-1] < max_len else x for x in mix_db_list]
    vocals_db_list = [torch.cat([x, torch.zeros((x.shape[0], max_len - x.shape[1]))], dim=-1) if x.shape[-1] < max_len else x for x in vocals_db_list]
    return torch.stack(mix_db_list, 0), torch.stack(vocals_db_list, 0)

def train(model, dataloader, optimizer, criterion, device):
    """
    Entrena el modelo por una época.
    """
    model.train()
    running_loss = 0.0
    for mix_db, vocals_db in tqdm(dataloader, desc="Training"):
        mix_db, vocals_db = mix_db.to(device), vocals_db.to(device)
        optimizer.zero_grad()
        outputs = model(mix_db)
        loss = criterion(outputs, vocals_db)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

#%% DATASET Y DATALOADER

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
        mix, _ = librosa.load(self.audio_files[idx], sr=self.sr)
        vocals, _ = librosa.load(self.vocal_files[idx], sr=self.sr)

        if len(mix) > self.sr * self.cut:
            start = (len(mix) - self.sr * self.cut) // 2
            end = start + self.sr * self.cut
            mix = mix[start:end]
            vocals = vocals[start:end]

        mix_mel = librosa.feature.melspectrogram(y=mix, sr=self.sr, n_mels=self.n_mels)
        vocals_mel = librosa.feature.melspectrogram(y=vocals, sr=self.sr, n_mels=self.n_mels)

        mix_db = librosa.power_to_db(mix_mel, ref=np.max)
        vocals_db = librosa.power_to_db(vocals_mel, ref=np.max)

        min_db, max_db = -80.0, 0.0
        vocals_db = (vocals_db - min_db) / (max_db - min_db)

        max_len = min(self.max_len, mix_db.shape[-1])
        if mix_db.shape[-1] < max_len:
            pad_width = max_len - mix_db.shape[-1]
            mix_db = np.pad(mix_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=1e-5)
            vocals_db = np.pad(vocals_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=1e-5)
        else:
            mix_db = mix_db[:, :max_len]
            vocals_db = vocals_db[:, :max_len]

        mix_db = torch.tensor(mix_db, dtype=torch.float32)
        vocals_db = torch.tensor(vocals_db, dtype=torch.float32)

        return mix_db, vocals_db

audio_files, vocal_files = load_data(data_dir)
dataset = AudioDataset(audio_files, vocal_files)
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)

#%% MODELO RNN
class VocalSeparationRNN(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, output_size=128, bidirectional=True):
        super(VocalSeparationRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output.transpose(1, 2)

#%% ENTRENAMIENTO
model = VocalSeparationRNN().to(device)
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
plt.savefig(f'plots/rnn_vocal_separation_loss_{epochs}epochs_{lr}lr_L1.png')
torch.save(model.state_dict(), f'models/rnn_vocal_separation_model_{epochs}epochs_{lr}lr_L1.pth')