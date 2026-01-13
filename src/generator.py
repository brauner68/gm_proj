import torch
import os
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

# Import your modules
from src.model import TimeConditionedUnet, ConcatConditionedUnet
from src.vocoder import BigVGAN_Vocoder
from src.denoising import denoise_audio_tensor, denoise_tensor_via_nlm

class DiffusionGenerator:
    def __init__(self, config, checkpoint_path):
        """
        Args:
            config (dict): The configuration dict used for training.
            checkpoint_path (str): Path to the trained .pt model file.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        print(f"⏳ Initializing Generator on {self.device}...")

        # 1. Setup Label Map
        self.label_map = self._create_label_map(config['selected_families'])
        self.num_classes = len(self.label_map)

        # 2. Initialize Vocoder
        self.vocoder = BigVGAN_Vocoder(device=self.device)

        # 3. Initialize Model
        # Add 1 for the 'null' token
        if self.config['conditioning'] == 'time':
            self.model = TimeConditionedUnet(
                num_classes=self.num_classes + 1,
                num_pitches=129,
                T=config['T_target'],
                use_pitch=config['use_pitch']
            ).to(self.device)
        else:
            self.model = ConcatConditionedUnet(
                num_classes=self.num_classes + 1,
                T=config['T_target'],
                num_pitches=129,
                use_pitch=config['use_pitch']
            ).to(self.device)

        # 4. Load Weights
        self._load_checkpoint()

        # 5. Initialize Scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config['num_train_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            beta_schedule=config['beta_schedule']
        )

        print("✅ Generator Ready.")

    def _create_label_map(self, selected_families):
        if selected_families:
            return {name: i for i, name in enumerate(selected_families)}
        return {name: i for i, name in enumerate(
            ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
             'organ', 'reed', 'string', 'synth_lead', 'vocal']
        )}

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"   Loaded weights from {self.checkpoint_path}")

    @torch.no_grad()
    def generate(self, pitch=60, samples_per_class=1, output_dir=None):
        """
        Generates audio for every class in the config using batch processing.
        """
        save_dir = output_dir if output_dir else os.path.join(self.config['output_dir'], "generated")
        os.makedirs(save_dir, exist_ok=True)

        print(f"🎹 Generating {samples_per_class} samples per class...")
        print(f"📂 Output: {save_dir}")

        results = {}
        use_pitch = self.config['use_pitch']

        # ---- 1) Build ONE big batch for ALL instruments ----
        instrument_names = list(self.label_map.keys())
        label_indices = list(self.label_map.values())
        n_classes = len(instrument_names)

        # Total batch size = classes * samples_per_class
        B = n_classes * samples_per_class

        # Labels: repeat each class label samples_per_class times (grouped per instrument)
        labels = torch.tensor(label_indices, device=self.device, dtype=torch.long)
        labels = labels.repeat_interleave(samples_per_class)  # [B]

        # Pitches (optional): same pitch for the entire batch
        if use_pitch:
            pitches = torch.full((B,), pitch, device=self.device, dtype=torch.long)
        else:
            pitches = None

        # Latents: [B, 1, 80, T]
        latents = torch.randn(
            (B, 1, 80, self.config['T_target']),
            device=self.device
        )

        # ---- 2) Diffusion Loop ----
        for t in tqdm(self.noise_scheduler.timesteps, desc="Gen: all instruments", leave=False):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(latents, t_batch, labels, pitches)
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # --- DENOISING BLOCK ---
        method = self.config.get('denoise_method', None)
        strength = self.config.get('denoise_strength', 0)

        if method == 'spectral':
            print(f"   🧹 Applying Spectral Gating (Strength: {strength})")
            latents = denoise_audio_tensor(latents, strength=strength)

        elif method == 'nlm':
            print(f"   🧹 Applying NLM Denoising (Strength: {strength})")
            # Note: NLM is slower as it moves data to CPU and back
            latents = denoise_tensor_via_nlm(latents, strength=int(strength))

        # ---- 3) Batch Decoding ----
        specs_batch = latents.squeeze(1)  # [B, 80, T]
        audio_batch = self.vocoder.decode(specs_batch)  # typically [B, 1, T_audio] (depends on your vocoder)

        # ---- 4) Save Individually + Build results dict ----
        idx = 0
        for instrument_name in instrument_names:
            results[instrument_name] = []
            for i in range(samples_per_class):
                filename = f"{instrument_name}_{i + 1}.wav"
                path = os.path.join(save_dir, filename)

                self.vocoder.save_audio(audio_batch[idx], path)
                results[instrument_name].append(audio_batch[idx].cpu())
                idx += 1

        print("✨ Generation Complete!")
        return results

