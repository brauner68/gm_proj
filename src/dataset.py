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
        Loads files and equalizes them so every instrument has the same number of samples.
        If max_samples is set, it treats it as the limit PER CLASS.
        """
        all_files = glob.glob(os.path.join(data_path, 'audio', '*.wav'))
        if not all_files:
            all_files = glob.glob(os.path.join(data_path, 'nsynth_valid', 'audio', '*.wav'))

        # 1. Group files by Instrument
        # storage = { 'guitar': [file1, file2], 'flute': [file3] ... }
        storage = {name: [] for name in self.label_map.keys()}

        print("🔍 Scanning and sorting files...")
        for p in all_files:
            filename = os.path.basename(p)
            parts = filename.split('_')

            # Check Source (Acoustic only)
            if len(parts) > 1 and parts[0] == 'synth' and parts[1] == 'lead':
                source = parts[2]
            else:
                source = parts[1]

            if source != 'acoustic':
                continue

            # Check Instrument
            name, _ = self._get_info_from_path(p)
            if name in self.label_map:
                storage[name].append(p)

        # 2. Determine the "Common Count"
        # We find the instrument with the FEWEST files to set the baseline.
        # If any instrument has 0 files, we must ignore it or the limit becomes 0.
        valid_counts = [len(files) for files in storage.values() if len(files) > 0]

        if not valid_counts:
            print("❌ No acoustic files found!")
            return []

        min_available = min(valid_counts)

        # If user provided a limit (e.g. 500), we take min(500, real_min)
        if max_samples:
            target_count = min(min_available, max_samples)
        else:
            target_count = min_available

        print(f"⚖️  Balancing Dataset: Limiting all classes to {target_count} samples.")

        # 3. Flatten into final list
        valid_files = []
        counts = {}

        for name, files in storage.items():
            if len(files) > 0:
                # Take exactly 'target_count' samples
                # We shuffle first to get random ones, not just the first 500 alphabetically
                import random
                random.shuffle(files)
                selected = files[:target_count]

                valid_files.extend(selected)
                counts[name] = len(selected)
            else:
                counts[name] = 0

        # 4. Print Statistics
        print("\n📊 Dataset Statistics (Balanced & Acoustic):")
        print(f"{'Instrument':<15} | {'Count':<10}")
        print("-" * 30)
        total = 0
        for name, count in counts.items():
            if count > 0:
                print(f"{name:<15} | {count:<10}")
                total += count
        print("-" * 30)
        print(f"{'TOTAL':<15} | {total:<10}\n")

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
    def __init__(self, data_path, T_target=160, max_samples=None, selected_families=None, equalize_data=False):
        """
        Specialized Dataset for BigVGAN (22kHz, 80 Mels, 256 Hop).
        """
        self.data_path = data_path
        self.label_map = self._create_label_map(selected_families)
        self.files = self._load_and_filter_files(data_path, max_samples, equalize=equalize_data)

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

    def _load_and_filter_files(self, data_path, max_samples, equalize):
        """
        Loads files with two modes:
        1. Equalize=True:  Every class gets exactly min(count, max_samples) files.
        2. Equalize=False: We just take the first max_samples files total (imbalanced).
        """
        all_files = glob.glob(os.path.join(data_path, 'audio', '*.wav'))
        if not all_files:
            all_files = glob.glob(os.path.join(data_path, 'nsynth_valid', 'audio', '*.wav'))

        # --- Step 1: Filter for Acoustic & Selected Families ---
        storage = {name: [] for name in self.label_map.keys()}

        print(f"🔍 Scanning files (Equalize={equalize})...")
        for p in all_files:
            # Parse filename
            filename = os.path.basename(p)
            parts = filename.split('_')

            # Check Source (Acoustic only)
            if len(parts) > 1 and parts[0] == 'synth' and parts[1] == 'lead':
                source = parts[2]
            else:
                source = parts[1]

            if source != 'acoustic':
                continue

            # Check Instrument Family
            name, _ = self._get_info_from_path(p)
            if name in self.label_map:
                storage[name].append(p)

        # --- Step 2: Select Files based on Mode ---
        valid_files = []
        counts = {}

        if equalize:
            # --- MODE A: BALANCED ---
            # Find the lowest count (ignoring empty classes if you want, or stopping if 0)
            valid_counts = [len(f) for f in storage.values() if len(f) > 0]
            if not valid_counts: return []

            min_available = min(valid_counts)

            # Limit per class
            target_count = min(min_available, max_samples) if max_samples else min_available
            print(f"⚖️  Balancing: Limiting all classes to {target_count} samples.")

            for name, files in storage.items():
                if len(files) > 0:
                    import random
                    random.shuffle(files)
                    selected = files[:target_count]
                    valid_files.extend(selected)
                    counts[name] = len(selected)
                else:
                    counts[name] = 0
        else:
            # --- MODE B: IMBALANCED (Standard) ---
            # Just flatten everything into one list
            for name, files in storage.items():
                valid_files.extend(files)
                counts[name] = len(files)

            # Shuffle globally
            import random
            random.shuffle(valid_files)

            # Apply Total Limit
            if max_samples:
                print(f"✂️  Limiting TOTAL dataset to {max_samples} samples.")
                valid_files = valid_files[:max_samples]

                # Re-count after cutting (approximate)
                counts = {name: 0 for name in self.label_map}
                for p in valid_files:
                    n, _ = self._get_info_from_path(p)
                    counts[n] += 1

        # --- Step 3: Print Statistics ---
        print("\n📊 Dataset Statistics:")
        print(f"{'Instrument':<15} | {'Count':<10}")
        print("-" * 30)
        total = 0
        for name, count in counts.items():
            if count > 0:
                print(f"{name:<15} | {count:<10}")
                total += count
        print("-" * 30)
        print(f"{'TOTAL':<15} | {total:<10}\n")

        return valid_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        # Get Info
        instrument_name, pitch_val = self._get_info_from_path(path)
        label = torch.tensor(self.label_map[instrument_name], dtype=torch.long)
        pitch = torch.tensor(pitch_val, dtype=torch.long)

        # 1. Get raw spectrogram [1, 80, T]
        # Range is approx [-11.5, 2.5]
        spec = self.vocoder.encode(path, self.T_target)

        # 2. Normalize to [-1, 1]
        # We assume min=-12.0 and max=3.0 to be safe
        min_db = -12.0
        max_db = 3.0

        # Clip to be safe
        spec = torch.clamp(spec, min_db, max_db)

        # Scale to [0, 1]
        spec = (spec - min_db) / (max_db - min_db)

        # Scale to [-1, 1]
        spec = spec * 2.0 - 1.0

        return spec, label, pitch
