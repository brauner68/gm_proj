import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as VT
import os
import glob
from torch.utils.data import Dataset


class NSynthDataset(Dataset):
    def __init__(self, data_path, target_size=(64, 64), max_samples=None, selected_families=None):
        """
        Args:
            selected_families (list[str]): e.g. ['guitar', 'brass']. If None, uses all 11.
        """
        self.data_path = data_path
        self.target_size = target_size
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
            sample_rate=self.sample_rate, n_fft=1024, hop_length=512, n_mels=target_size[0]
        )
        self.resize_transform = VT.Resize(target_size)

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
        spec = self.mel_transform(waveform)
        spec = torch.log(spec + 1e-9)
        image = self.resize_transform(spec)

        # Normalize
        min_val = image.min()
        max_val = image.max()
        if (max_val - min_val) > 1.0:
            image = (image - min_val) / (max_val - min_val + 1e-5)
            image = image * 2 - 1
        else:
            image = torch.ones_like(image) * -1

        return image, label