import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as VT
import os
import glob
from torch.utils.data import Dataset

from src.vocoder import BigVGAN_Vocoder


class NSynthDataset(Dataset):
    def __init__(self, data_path, target_size=(64, 64), hop_length=512, max_samples=None, selected_families=None):
        """
        Args:
            selected_families (list[str]): e.g. ['guitar', 'brass']. If None, uses all 11.
        """
        self.data_path = data_path
        self.target_size = target_size
        self.hop_length = hop_length
        self.target_samples = hop_length * target_size[1]
        self.sample_rate = 16000

        # 1. Setup the Map (Dynamic Re-mapping)
        self.label_map = self._create_label_map(selected_families)
        print(f"Label Map: {self.label_map}")

        # 2. Find and Filter Files
        self.files = self._load_and_filter_files(data_path, max_samples)

        if len(self.files) == 0:
            raise RuntimeError(f"No audio files found in {data_path} matching {list(self.label_map.keys())}")

        print(f"Dataset initialized. {len(self.files)} files selected.")

        # 3. Define Transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024, hop_length=hop_length, n_mels=target_size[0]
        )

    def _create_label_map(self, selected_families):
        """
        Creates a dictionary mapping instrument names to integers 0..N
        """
        if selected_families:
            # User wants specific instruments: Map them to 0, 1, 2...
            return {name: i for i, name in enumerate(selected_families)}
        else:
            # Default: Use all 11 standard NSynth families
            default_families = [
                'bass', 'brass', 'flute', 'guitar', 'keyboard',
                'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'
            ]
            return {name: i for i, name in enumerate(default_families)}

    def _get_instrument_from_path(self, path):
        """
        Extracts 'guitar' from 'guitar_acoustic_001.wav'
        """
        filename = os.path.basename(path)
        parts = filename.split('_')
        # Special case for synth_lead which has an underscore
        if len(parts) > 1 and parts[0] == 'synth' and parts[1] == 'lead':
            return "synth_lead"
        return parts[0]

    def _load_and_filter_files(self, data_path, max_samples):
        """
        Loads ALL files, then removes the ones we don't want.
        """
        # A. Find all wav files (check both possible folder structures)
        all_files = glob.glob(os.path.join(data_path, 'audio', '*.wav'))

        # B. Filter: Only keep files that are in our label_map
        valid_files = []
        for path in all_files:
            instrument = self._get_instrument_from_path(path)
            if instrument in self.label_map:
                valid_files.append(path)

        # C. Apply Max Samples (for debugging)
        if max_samples:
            valid_files = valid_files[:max_samples]

        return valid_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        # --- Load Audio ---
        try:
            waveform, sr = torchaudio.load(path)
        except Exception:
            import soundfile as sf
            audio_data, sr = sf.read(path)
            waveform = torch.from_numpy(audio_data).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)

        # --- Get Label ---
        instrument_name = self._get_instrument_from_path(path)
        # We don't need .get() or checks here because we already filtered self.files!
        label_int = self.label_map[instrument_name]
        label = torch.tensor(label_int, dtype=torch.long)

        # --- Process Image ---
        # Crop to fit the net
        if waveform.shape[1] > self.target_samples:
            waveform = waveform[:, :self.target_samples]
        else:
            # Fallback: if file is too short (rare), pad it
            pad_amt = self.target_samples - waveform.shape[1]
            import torch.nn.functional as F
            waveform = F.pad(waveform, (0, pad_amt))
        spec = self.mel_transform(waveform)
        spec = torch.log(spec + 1e-9)

        # Normalize
        min_val = spec.min()
        max_val = spec.max()
        if (max_val - min_val) > 1.0:
            spec = (spec - min_val) / (max_val - min_val + 1e-5)
            spec = spec * 2 - 1
        else:
            spec = torch.ones_like(spec) * -1

        return spec, label


class BigVGAN_NSynthDataset(Dataset):
    def __init__(self, data_path, T_target=160, max_samples=None, selected_families=None):
        """
        Specialized Dataset for BigVGAN (22kHz, 80 Mels, 256 Hop).
        """
        self.data_path = data_path
        self.label_map = self._create_label_map(selected_families)
        self.files = self._load_and_filter_files(data_path, max_samples)

        if len(self.files) == 0:
            raise RuntimeError(f"No files found in {data_path}")

        print(f"BigVGAN NSynth Dataset: {len(self.files)} files")

        # Load BigVGAN vocoder class
        self.vocoder = BigVGAN_Vocoder(device='cpu')
        self.T_target = T_target # width of mel spec

    # --- Reuse your existing helper methods ---
    def _create_label_map(self, selected_families):
        if selected_families: return {name: i for i, name in enumerate(selected_families)}
        return {name: i for i, name in enumerate(
            ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead',
             'vocal'])}

    def _get_info_from_path(self, path):
        """
        Extracts instrument name AND pitch from filename.
        Format: instrument_family_source_id_pitch-velocity.wav
        Example: guitar_acoustic_001-060-127.wav
        """
        filename = os.path.basename(path)

        # 1. Get Instrument Name
        parts = filename.split('_')
        if len(parts) > 1 and parts[0] == 'synth' and parts[1] == 'lead':
            inst_name = "synth_lead"
        else:
            inst_name = parts[0]

        # 2. Get Pitch
        # The pitch is usually in the last segment after the last '_'
        # Example part: "001-060-127.wav"
        last_part = parts[-1]
        try:
            # Split "001-060-127.wav" by "-" -> ["001", "060", "127.wav"]
            meta_parts = last_part.split('-')
            # Pitch is the middle number (MIDI 0-127)
            pitch = int(meta_parts[1])
        except (IndexError, ValueError):
            # Fallback if filename format is weird
            print(f"Warning: Could not parse pitch from {filename}, defaulting to 60 (Middle C)")
            pitch = 60

        return inst_name, pitch

    def _load_and_filter_files(self, data_path, max_samples):
        all_files = glob.glob(os.path.join(data_path, 'audio', '*.wav'))
        if not all_files:
            all_files = glob.glob(os.path.join(data_path, 'nsynth_valid', 'audio', '*.wav'))

        valid_files = []
        for p in all_files:
            name, _ = self._get_info_from_path(p)
            if name in self.label_map:
                valid_files.append(p)

        if max_samples: valid_files = valid_files[:max_samples]
        return valid_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        # Get Label
        instrument_name, pitch_val = self._get_info_from_path(path)
        label = torch.tensor(self.label_map[instrument_name], dtype=torch.long)
        pitch = torch.tensor(pitch_val, dtype=torch.long)  # MIDI pitch is 0-127

        # Get spectrogram
        spec = self.vocoder.encode(path, self.T_target) # [1, 80, T]

        return spec, label, pitch
