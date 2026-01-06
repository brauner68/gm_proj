import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import os


class Vocoder:
    def __init__(self, device='cpu', n_mels=64, hop_length=512):
        self.device = device
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = hop_length
        self.n_mels = n_mels  # Must match dataset.py
        self.amp = 8.0

        # 1. Inverse Mel Transform: (Mel -> Linear Spectrogram)
        # We need to recover the linear frequencies from the Mel bands.
        self.inverse_mel = T.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,  # 513 bins
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=0.0,
            # f_max=None defaults to sample_rate // 2
        ).to(device)

        # 2. Griffin-Lim: (Linear Spectrogram -> Audio Waveform)
        # Reconstructs phase iteratively.
        self.griffin_lim = T.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=1.0,  # We will feed it Magnitude, not Power
            n_iter=128  # More iterations = better quality, slower
        ).to(device)

    def decode(self, mel_spectrogram):
        """
        Args:
            mel_spectrogram: Tensor of shape [Channels, Height, Width]
                             Values should be in range [-1, 1] (from Diffusion)
        Returns:
            waveform: Tensor of shape [1, Time]
        """
        spec = mel_spectrogram.to(self.device)

        # Remove Batch dimension if present
        if spec.dim() == 4:
            spec = spec.squeeze(0)  # [1, 64, 64] -> [64, 64]

        # Remove Channel dimension if present
        if spec.dim() == 3:
            spec = spec.squeeze(0)  # [1, 64, 64] -> [64, 64]

        # --- Step A: Denormalize ---
        # The model outputs [-1, 1]. We need to map this back to linear magnitude.

        # 1. Shift [-1, 1] -> [0, 1]
        spec = (spec + 1) / 2.0

        # 2. Scale up (Heuristic)
        # Log-mels are usually roughly in range -10 to 5.
        # We multiply by a gain factor to restore dynamic range before exp.
        # 5.0 to 10.0 is a safe heuristic for audible volume.
        spec = spec * self.amp

        # 3. Inverse Log (Exp)
        spec = torch.exp(spec)

        # --- Step B: Inverse Mel ---
        # Input: [n_mels, time] -> Output: [n_stft, time]
        linear_spec = self.inverse_mel(spec)

        # --- Step C: Griffin-Lim ---
        # Input: [n_stft, time] -> Output: [samples]
        waveform = self.griffin_lim(linear_spec)

        return waveform.unsqueeze(0)  # Add channel dim back: [1, Time]

    def save_audio(self, waveform, path):
        """
        Saves the tensor waveform to a .wav file.
        """
        # Ensure CPU and Numpy
        wav_numpy = waveform.squeeze().detach().cpu().numpy()

        # Make directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        sf.write(path, wav_numpy, self.sample_rate)
        print(f"Saved audio to {path}")


# Make sure to install: pip install bigvgan
try:
    import bigvgan
except ImportError:
    print("⚠️ BigVGAN not found. Please run: pip install bigvgan")


class BigVGAN_Vocoder:
    def __init__(self, device='cpu'):
        self.device = device
        print("⏳ Loading BigVGAN (Universal 22kHz)...")

        # Auto-download the model from Hugging Face
        self.model = bigvgan.BigVGAN.from_pretrained(
            'nvidia/bigvgan_v2_22khz_80band_256x',
            use_cuda_kernel=False
        ).to(self.device)

        self.model.remove_weight_norm()
        self.model.eval()
        print("✅ BigVGAN Loaded.")

    @torch.no_grad()
    def decode(self, mel_spectrogram):
        """
        Args:
            mel_spectrogram: [Batch, 1, 80, Time] - Values in [-1, 1]
        Returns:
            audio: [Batch, 1, Time]
        """
        spec = mel_spectrogram.to(self.device)

        # 1. Shape Check: Needs to be [Batch, 80, Time]
        if spec.dim() == 4:
            spec = spec.squeeze(1)  # Remove channel dim [B, 1, 80, T] -> [B, 80, T]
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)  # Add batch dim [80, T] -> [1, 80, T]

        # 2. Denormalize (Crucial Step)
        # The Diffusion model outputs [-1, 1].
        # BigVGAN expects raw log-mel values (roughly -11 to +4).
        # We assume the user normalized nicely, so we stretch it back.
        # Heuristic: Mult by 6 to get range 12, shift down by 4.
        # Result range: approx -10 to +2
        spec = (spec * 6.0) - 4.0

        # 3. Generate Audio
        audio = self.model(spec)

        return audio  # Returns [Batch, 1, Time]

    def save_audio(self, waveform, path):
        import soundfile as sf
        wav_numpy = waveform.squeeze().cpu().numpy()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Note: BigVGAN is 22050Hz!
        sf.write(path, wav_numpy, 22050)
        print(f"Saved audio to {path}")