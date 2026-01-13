import os
# --- SMART CACHE SETUP ---
# 1. Define your personal persistent path
cluster_base_path = "/gpfs0/bgu-br/users/nitaype/gm_proj"
custom_cache_path = os.path.join(cluster_base_path, ".cache_hf")

# 2. Check: Are we on the cluster? (Does the base folder exist?)
if os.path.exists(cluster_base_path):
    # Yes -> Force Hugging Face to use your folder
    os.environ['HF_HOME'] = custom_cache_path
    print(f"✅ Detected Cluster Environment. Using persistent cache: {custom_cache_path}")
else:
    # No -> We are likely on Colab or another machine
    print("⚠️ Standard Environment detected. Using default cache (downloads may occur).")

from __future__ import annotations
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf



class Vocoder:
    def __init__(self, device='cpu', n_mels=64, hop_length=512):
        self.device = device
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = hop_length
        self.n_mels = n_mels  # Must match dataset.py

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
            n_iter=32  # More iterations = better quality, slower
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
        # Log-mels are usually roughly in range -6 to 6.
        # We multiply by a gain factor to restore dynamic range before exp.
        spec = spec * 12 - 6.0

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



import bigvgan
from bigvgan.meldataset import get_mel_spectrogram
import librosa

class BigVGAN_Vocoder:
    def __init__(self, device='cpu'):
        self.device = device
        print("⏳ Loading BigVGAN (Universal 22kHz)...")

        # Auto-download the model from Hugging Face
        self.model = bigvgan.BigVGAN.from_pretrained(
            'nvidia/bigvgan_v2_22khz_80band_fmax8k_256x',
            use_cuda_kernel=False,
        ).to(self.device)

        self.model.remove_weight_norm()
        self.model.eval()
        print("✅ BigVGAN Loaded.")

    @torch.inference_mode()
    def encode(self, wav_path, T_target=None):
        wav, sr = librosa.load(wav_path, sr=self.model.h.sampling_rate,
                               mono=True)  # wav is np.ndarray with shape [T_time] and values in [-1, 1]
        if T_target is not None:
            N = T_target * self.model.h.hop_size
            wav = wav[:N]
        wav = torch.FloatTensor(wav).unsqueeze(0)  # wav is FloatTensor with shape [B(1), T_time]
        wav = torch.clamp(wav, -1.0,1.0).to(self.device)
        mel = get_mel_spectrogram(wav, self.model.h).to(self.device)  # mel is FloatTensor with shape [B(1), C_mel, T_frame]
        return mel

    @torch.inference_mode()
    def decode(self, mel_spectrogram):
        """
        Input: Spectrogram Batch [Batch, 80, T]
        Output: Audio Batch [Batch, 1, T_audio]
        """
        mel = mel_spectrogram.to(self.device)
        wav_gen = self.model(mel)
        return wav_gen.cpu()

    def save_audio(self, waveform, path):
        """
        Input: Single waveform tensor [1, T] or [T]
        """
        import soundfile as sf
        # Ensure it's on CPU and squeeze channel/batch dims for saving
        wav_numpy = waveform.squeeze().cpu().numpy()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, wav_numpy, 22050)
        print(f"Saved audio to {path}")