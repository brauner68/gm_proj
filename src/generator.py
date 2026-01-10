import torch
import os
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

# Import your modules
from src.model import TimeConditionedUnet
from src.vocoder import BigVGAN_Vocoder


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
        self.model = TimeConditionedUnet(
            num_classes=self.num_classes + 1,
            T=config['T_target']
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
    def generate(self, samples_per_class=1, output_dir=None):
        """
        Generates audio for every class in the config using batch processing.
        """
        save_dir = output_dir if output_dir else os.path.join(self.config['output_dir'], "generated")
        os.makedirs(save_dir, exist_ok=True)

        print(f"🎹 Generating {samples_per_class} samples per class...")
        print(f"📂 Output: {save_dir}")

        # Dictionary to store results: {'guitar': [wav1, wav2], 'flute': ...}
        results = {}

        for instrument_name, label_idx in self.label_map.items():

            # --- 1. Prepare Batch ---
            # Labels: [samples_per_class]
            labels = torch.full((samples_per_class,), label_idx, device=self.device, dtype=torch.long)

            # Latents: [samples_per_class, 1, 80, T]
            latents = torch.randn(
                (samples_per_class, 1, 80, self.config['T_target']),
                device=self.device
            )

            # --- 2. Diffusion Loop ---
            desc = f"Gen: {instrument_name}"
            for t in tqdm(self.noise_scheduler.timesteps, desc=desc, leave=False):
                # Expand time to batch
                t_batch = torch.full((samples_per_class,), t, device=self.device, dtype=torch.long)

                # Predict & Step
                noise_pred = self.model(latents, t_batch, labels)
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

            # --- 3. Batch Decoding ---
            # Model output is [B, 1, 80, T]. Vocoder expects [B, 80, T].
            specs_batch = latents.squeeze(1)

            # Decode the ENTIRE batch at once
            # Returns [B, 1, T_audio]
            audio_batch = self.vocoder.decode(specs_batch)

            # --- 4. Save Individually ---
            # Store in results dict
            results[instrument_name] = []
            for i in range(samples_per_class):
                filename = f"{instrument_name}_{i + 1}.wav"
                path = os.path.join(save_dir, filename)

                # Pass single item [1, T_audio] to save
                self.vocoder.save_audio(audio_batch[i], path)

                # Add to results (keep on CPU for display)
                results[instrument_name].append(audio_batch[i].cpu())

        print("✨ Generation Complete!")
        return results
