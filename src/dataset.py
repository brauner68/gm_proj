import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as VT
import os
import glob
from torch.utils.data import Dataset


class NSynthDataset(Dataset):
    def __init__(self, data_path, target_size=(64, 64), max_samples=None):
        self.data_path = data_path

        # NOTE: data_path must be the UNZIPPED folder, not the .tar.gz file
        self.files = glob.glob(os.path.join(data_path, 'audio', '*.wav'))

        # If that fails, try looking directly in 'audio' (depends on how it was unzipped)
        if not self.files:
            self.files = glob.glob(os.path.join(data_path, 'audio', '*.wav'))

        if len(self.files) == 0:
            raise RuntimeError(f"❌ No audio files found in {data_path}. Did you unzip the .tar.gz file?")

        if max_samples:
            self.files = self.files[:max_samples]

        self.target_size = target_size
        self.sample_rate = 16000

        # 1. Audio Transform (Torchaudio)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=target_size[0]
        )

        # 2. Image Transform (Torchvision) <--- FIXED LINE
        self.resize_transform = VT.Resize(target_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # ... (Rest of the logic is the same as before) ...
        path = self.files[idx]

        try:
            waveform, sr = torchaudio.load(path)
        except Exception:
            import soundfile as sf
            audio_data, sr = sf.read(path)
            waveform = torch.from_numpy(audio_data).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)

        filename = os.path.basename(path)
        parts = filename.split('_')

        if len(parts) > 1 and parts[0] == 'synth' and parts[1] == 'lead':
            instrument_name = "synth_lead"
        else:
            instrument_name = parts[0]

        spec = self.mel_transform(waveform)
        spec = torch.log(spec + 1e-9)

        # Apply Resize using the Vision library
        image = self.resize_transform(spec)

        min_val = image.min()
        max_val = image.max()
        if (max_val - min_val) > 1.0:
            image = (image - min_val) / (max_val - min_val + 1e-5)
            image = image * 2 - 1
        else: # if for some reason the audio is silence
            image = torch.ones_like(image) * -1

        return image, instrument_name