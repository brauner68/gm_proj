import torch
import torch.nn as nn
import torchaudio
import os
import glob
from torch.utils.data import Dataset


class NSynthDataset(Dataset):
    def __init__(self, data_path, target_size=(64, 64), max_samples=None):
        """
        Args:
            data_path (str): Path to the UNZIPPED folder (e.g., /content/data/nsynth-valid)
            target_size (tuple): Desired spectrogram size (height, width)
            max_samples (int): If set, only load first N samples (for debugging)
        """
        self.data_path = data_path
        # NSynth audio files are usually inside an 'audio' subfolder
        self.files = glob.glob(os.path.join(data_path, 'audio', '*.wav'))

        if max_samples:
            self.files = self.files[:max_samples]

        self.target_size = target_size

        # Audio Settings (Must match what we tested in the notebook)
        self.sample_rate = 16000
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=target_size[0]  # Height
        )
        self.resize_transform = torchaudio.transforms.Resize(target_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        # 1. Load Audio
        # Torchaudio loads as [Channels, Time]. We assume Mono [1, Time]
        waveform, sr = torchaudio.load(path)

        # 2. Extract Labels from Filename
        # NSynth Format: instrument_family_str_source-qual-pitch-vel.wav
        # Example: guitar_acoustic_001-060-100.wav
        filename = os.path.basename(path)
        parts = filename.split('_')

        # Extract Instrument Family (Simple string parsing)
        # Note: In a real project, we would map these strings to Integers (0-10) using a dictionary.
        # For now, let's just return the filename so we can debug.
        instrument_name = parts[0]

        # Extract Pitch (It's usually in the last part)
        # We will handle string-to-int mapping later.

        # 3. Process to Spectrogram
        spec = self.mel_transform(waveform)

        # Log Scale (Important for audio! makes it look like an image)
        spec = torch.log(spec + 1e-9)

        # Resize to fixed square
        image = self.resize_transform(spec)

        # Normalize to [-1, 1] range (Crucial for Diffusion Models)
        # (Assuming log-spec values are roughly -10 to 4)
        # A simple min-max norm per image is a safe start:
        min_val = image.min()
        max_val = image.max()
        image = (image - min_val) / (max_val - min_val + 1e-5)  # [0, 1]
        image = image * 2 - 1  # [-1, 1]

        return image, instrument_name
