# KNOWLEDGE DISTILLATION TRAINING SCRIPT

"""
This script is used to train the student model using the knowledge distillation technique. The teacher model is the pretrained open-unmix model. 
The student model is a simple CNN model. 
"""

#%% IMPORTS and SETUP
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
import torchaudio
import pandas as pd

warnings.filterwarnings("ignore")

os.chdir("../")
data_dir = os.getcwd() + "/dataset/train"
test_dir = os.getcwd() + "/dataset/test"

# loading umxhq four target separator
separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq')

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

def spectral_convergence_loss(pred_spectrogram, target_spectrogram):
    """
    Computes the Spectral Convergence Loss.
    Args:
    - pred_spectrogram (Tensor): Predicted spectrogram (magnitude).
    - target_spectrogram (Tensor): Target spectrogram (magnitude).
    
    Returns:
    - loss (Tensor): Spectral Convergence Loss.
    """
    numerator = torch.norm(target_spectrogram - pred_spectrogram, p='fro')
    denominator = torch.norm(target_spectrogram, p='fro')
    loss = numerator / (denominator + 1e-8)  # Add epsilon to prevent division by zero
    return loss

def l1_loss(pred_spectrogram, target_spectrogram):
    """
    Computes the L1 Loss.
    Args:
    - pred_spectrogram (Tensor): Predicted spectrogram (magnitude).
    - target_spectrogram (Tensor): Target spectrogram (magnitude).
    
    Returns:
    - loss (Tensor): L1 Loss.
    """
    return torch.mean(torch.abs(pred_spectrogram - target_spectrogram))

def combined_loss(pred_spectrogram, target_spectrogram, alpha=0.5, beta=0.5):
    """
    Computes the combined Spectral Convergence and L1 Loss.
    Args:
    - pred_spectrogram (Tensor): Predicted spectrogram.
    - target_spectrogram (Tensor): Target spectrogram.
    - alpha (float): Weight for Spectral Convergence Loss.
    - beta (float): Weight for L1 Loss.
    
    Returns:
    - loss (Tensor): Combined Loss.
    """
    sc_loss = spectral_convergence_loss(pred_spectrogram, target_spectrogram)
    l1 = l1_loss(pred_spectrogram, target_spectrogram)
    return alpha * sc_loss + beta * l1

def soft_loss(student_output, teacher_output):
    """
    Computes the soft loss for voice isolation.

    Args:
    - student_output (Tensor): Student's predicted spectrogram or mask.
    - teacher_output (Tensor): Teacher's predicted spectrogram or mask.

    Returns:
    - loss (Tensor): L1 or L2 loss between student and teacher outputs.
    """
    # Use L1 loss to emphasize the magnitude similarity
    return F.l1_loss(student_output, teacher_output)

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

        # to tensor
        mix = torch.tensor(mix)
        vocals = torch.tensor(vocals)

        return mix, vocals
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

audio_files, vocal_files = load_data(data_dir)

# Create dataset and dataloader
audio_files, vocal_files = split_songs(audio_files, vocal_files)
dataset = AudioDataset(audio_files, vocal_files)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

val_audio_files, val_vocal_files  = load_data(test_dir)
val_audio_files, val_vocal_files = split_songs(val_audio_files, val_vocal_files)
val_dataset = AudioDataset(val_audio_files, val_vocal_files)
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

# print an example of a batch of data
mix_db_batch, vocals_db_batch = next(iter(dataloader))

#%% BASELINE MODEL
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
    
    def forward(self, x): # x: [B, 1, S]
        
        # Compute the mel spectrogram
        x = x.unsqueeze(1)  # Add channel dimension [B, 1, 128, 1292]

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)  # Remove channel dimension [B, 128, 1292]
    
#%% DISTILLATION TRAINING

student = VocalSeparationBaseline().to(device)

# track metrics for plotting training curves:
train_losses = []
val_losses = []
epoch_steps = [] # used for plotting val loss at the correct x-position
metrics = [train_losses, val_losses, epoch_steps]

def train_distillation(student,
                teacher,
                train_loader,
                val_loader,
                num_epochs,
                lr,  # initial learning rate
                l2_reg=0, # L2 weight decay term (0 means no regularisation)
                gamma=1, # exponential LR decay term (1 means no scheduling)
               ):

    #freeze the teacher model so we do not update its weights
    for param in teacher.parameters():
      param.requires_grad = False

    # define loss and optimiser:
    hard_loss = nn.L1Loss()
    soft_loss = nn.L1Loss()
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=l2_reg)
    #opt = torch.optim.SGD(student.parameters(), lr = lr, momentum = 0.9)

    # learning rate scheduler:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    # track metrics for plotting training curves:
    train_losses = []
    val_losses = []
    epoch_steps = [] # used for plotting val loss at the correct x-position
    metrics = [train_losses, val_losses, epoch_steps]
    best_val_loss = np.inf
    min_db, max_db = -80.0, 0.0  # Decibel range

    steps_trained = 0
    for e in range(num_epochs):
        train_bar = tqdm(train_loader, ncols=100)
        epoch_train_losses, epoch_train_accs = [], []
        for batch in train_bar:
            x, y = batch
            x_student_list = []
            y_list = []
            for i in range(x.size(0)):
                x_student_sample = librosa.feature.melspectrogram(y=x[i].cpu().numpy(), sr=22050, n_mels=128)
                y_sample = librosa.feature.melspectrogram(y=y[i].cpu().numpy(), sr=22050, n_mels=128)

                x_student_sample = librosa.power_to_db(x_student_sample, ref=np.max)
                y_sample = librosa.power_to_db(y_sample, ref=np.max)
                y_sample = (y_sample - min_db) / (max_db - min_db)


                x_student_list.append(torch.tensor(x_student_sample).to(device))
                y_list.append(torch.tensor(y_sample).to(device))

            x_student = torch.stack(x_student_list).to(device)
            y = torch.stack(y_list).to(device)
            
            #x, y = x.to(device), y.to(device)

            #x_student = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=128).to(device)(x)
            # spectogram with librosa
            #x_student = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=128).to(device)(x)
            

            # target to spectogram
            #y = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=128).to(device)(y)   
            # Convert Mel spectrograms to dB scale
            #x_student = torchaudio.transforms.AmplitudeToDB().to(device)(x_student)
            #y = torchaudio.transforms.AmplitudeToDB().to(device)(y)

            ## Normalize spectrograms
            #mix_db = (mix_db - mix_db.mean()) / mix_db.std()
            #vocals_db = (vocals_db - vocals_db.mean()) / vocals_db.std()

            original_sample_rate = 22050

            opt.zero_grad()

            student_pred = student(x_student)

            #hard_loss_value = hard_loss(student_pred, y)

            if e > 50:
                # make stereo
                #current shape is [B, S], we need [B, C, S] where C is the number of channels, 2 for stereo
                x_stereo = x.unsqueeze(1).repeat(1, 2, 1).to(device)

                # make sure to resample the audio to models' sample rate, separator.sample_rate, if the two are different
                resampler = torchaudio.transforms.Resample(original_sample_rate, teacher.sample_rate).to(device)
                x_stereo = resampler(x_stereo)
            
                teacher_pred = teacher(x_stereo)
                
                out_resampler = torchaudio.transforms.Resample(teacher.sample_rate, original_sample_rate).to(device)
                teacher_pred = out_resampler(teacher_pred)

                # get vocals from the estimates
                teacher_pred = teacher_pred[:, 0, 1, :]

                # compute spectrogram
                #teacher_pred = torchaudio.transforms.MelSpectrogram(sample_rate=teacher.sample_rate, n_mels=128).to(device)(teacher_pred)
                #teacher_pred = torchaudio.transforms.AmplitudeToDB().to(device)(teacher_pred)

                teacher_pred_list = []
                for i in range(teacher_pred.size(0)):
                    teacher_pred_sample = librosa.feature.melspectrogram(y=teacher_pred[i].cpu().numpy(), sr=22050, n_mels=128)
                    teacher_pred_sample = librosa.power_to_db(teacher_pred_sample, ref=np.max)
                    teacher_pred_sample = (teacher_pred_sample - min_db) / (max_db - min_db)
    
                    teacher_pred_list.append(torch.tensor(teacher_pred_sample).to(device))

                teacher_pred = torch.stack(teacher_pred_list).to(device)
                
                soft_loss_value = soft_loss(student_pred, teacher_pred)

                #hard_loss_value = combined_loss(student_pred, y)
                hard_loss_value = hard_loss(student_pred, y)

                # weighted avg
                alpha = 0.5
                batch_loss = alpha * hard_loss_value + (1 - alpha) * soft_loss_value

                #print(f"Hard loss: {hard_loss_value.item()}, Soft loss: {soft_loss_value.item()}, Combined loss: {batch_loss.item()}")

            else:
                #batch_loss = combined_loss(student_pred, y)
                batch_loss = hard_loss(student_pred, y)
        
            batch_loss.backward()
            opt.step()

            # track loss and accuracy:
            epoch_train_losses.append(batch_loss.item())
            steps_trained += 1

            train_bar.set_description(f'E{e} loss: {batch_loss:.2f}')

        epoch_steps.append(steps_trained)
        scheduler.step()

        # record training metrics, by batch and over the epoch:
        train_losses.extend(epoch_train_losses)

        print(f'train loss: {np.mean(epoch_train_losses):.2f}')

        if val_loader is not None:
            # evaluate after each epoch::
            with torch.no_grad():
                batch_val_losses = []

                for batch in val_loader:
                    vx, vy = batch
                    vx_student_list = []
                    vy_list = []
                    for i in range(vx.size(0)):
                        vx_student_sample = librosa.feature.melspectrogram(y=vx[i].cpu().numpy(), sr=22050, n_mels=128)
                        vy_sample = librosa.feature.melspectrogram(y=vy[i].cpu().numpy(), sr=22050, n_mels=128)
        
                        vx_student_sample = librosa.power_to_db(vx_student_sample, ref=np.max)
                        vy_sample = librosa.power_to_db(vy_sample, ref=np.max)
                        vy_sample = (vy_sample - min_db) / (max_db - min_db)
        
        
                        vx_student_list.append(torch.tensor(vx_student_sample).to(device))
                        vy_list.append(torch.tensor(vy_sample).to(device))
        
                    vx_student = torch.stack(vx_student_list).to(device)
                    vy = torch.stack(vy_list).to(device)

                    vpred = student(vx_student)

                    #val_loss = combined_loss(vpred, vy)
                    val_loss = hard_loss(vpred, vy)

                    batch_val_losses.append(val_loss.item())
                val_losses.append(np.mean(batch_val_losses))

                # update best accuracy:
                if val_losses[-1] < best_val_loss:
                    best_val_loss = val_losses[-1]
                    torch.save(student.state_dict(), f'models/best_student_{lr}lr_{num_epochs}epochs.pth')

            print(f'val loss: {np.mean(batch_val_losses):.2f}')

    print(f'Best validation loss: {best_val_loss:.2%}')
    torch.save(student.state_dict(), f'models/KD_model_{lr}lr_{num_epochs}epochs.pth')
    return metrics

teacher = separator.to(device)
num_epochs = 100
l2_reg = 1e-5
gamma = 1
lr = 1e-4

metrics = train_distillation(student, teacher, train_loader= dataloader, val_loader = val_dataloader, num_epochs= num_epochs, lr= lr, l2_reg = l2_reg,gamma = gamma)

def training_plot(metrics,
      title=None, # optional figure title
      alpha=0.05, # smoothing parameter for train loss 
      ):

    train_losses, val_losses, epoch_steps = metrics

    fig, (loss_ax) = plt.subplots(1,1)

    ### plot loss:
    smooth_train_loss = pd.Series(train_losses).ewm(alpha=alpha).mean()
    steps = np.arange(0, len(train_losses))

    # train loss is plotted at every step:
    loss_ax.plot(steps, smooth_train_loss, 'b-', label='train loss')
    # but val loss is plotted at every epoch:
    loss_ax.plot(epoch_steps, val_losses, 'r-', label='val loss')

    loss_ax.legend(); loss_ax.set_xlabel('Training step'); loss_ax.set_ylabel('Loss')

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'plots/KD_loss_100epochs_1e-4lr_L1.png')
    #plt.show()

training_plot(metrics)
