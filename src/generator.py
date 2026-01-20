import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

# Import your modules
from src.model import TimeConditionedUnet, ConcatConditionedUnet
from src.vocoder import BigVGAN_Vocoder


class DiffusionGenerator:
    def __init__(self, config, checkpoint_path):
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
                num_pitches=129,
                T=config['T_target'],
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
        # Fallback default
        return {name: i for i, name in enumerate(
            ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead',
             'vocal'])}

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"   Loaded weights from {self.checkpoint_path}")

    @torch.no_grad()
    def generate(self, pitch=60, samples_per_class=1, output_dir=None, guidance_scale=3.0, plot=True):
        """
        Generates audio for every class in parallel.
        Args:
            guidance_scale (float): CFG scale. >1.0 strengthens class characteristics.
            plot (bool): If True, saves a PNG spectrogram for each generated file.
        """
        save_dir = output_dir if output_dir else os.path.join(self.config['output_dir'], "generated")
        os.makedirs(save_dir, exist_ok=True)

        use_pitch = self.config.get('use_pitch', True)
        null_class = self.num_classes
        null_pitch = 128

        # --- 1. Construct the Batch ---
        all_labels = []
        all_pitches = []
        file_metadata = []

        for name, label_idx in self.label_map.items():
            all_labels.extend([label_idx] * samples_per_class)
            if use_pitch:
                all_pitches.extend([pitch] * samples_per_class)

            for i in range(samples_per_class):
                file_metadata.append(f"{name}_{i + 1}")  # Store filename base (no extension)

        total_samples = len(all_labels)
        print(f"🎹 Generating {total_samples} samples in parallel (Scale={guidance_scale})...")

        # Tensors
        cond_labels = torch.tensor(all_labels, device=self.device, dtype=torch.long)
        cond_pitches = torch.tensor(all_pitches, device=self.device, dtype=torch.long) if use_pitch else None

        uncond_labels = torch.full((total_samples,), null_class, device=self.device, dtype=torch.long)
        uncond_pitches = torch.full((total_samples,), null_pitch, device=self.device,
                                    dtype=torch.long) if use_pitch else None

        # --- 2. Initialize Latents ---
        latents = torch.randn(
            (total_samples, 1, 80, self.config['T_target']),
            device=self.device
        )

        # --- 3. Diffusion Loop ---
        for t in tqdm(self.noise_scheduler.timesteps, desc="Sampling", leave=False):
            # Input for CFG (Double Batch)
            latent_input = torch.cat([latents] * 2)
            t_batch = torch.full((total_samples * 2,), t, device=self.device, dtype=torch.long)

            label_batch = torch.cat([cond_labels, uncond_labels])
            pitch_batch = torch.cat([cond_pitches, cond_pitches]) if use_pitch else None

            # Predict
            noise_pred = self.model(latent_input, t_batch, label_batch, pitch_batch)

            # Guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Rescale CFG (Fix noise power)
            std_cond = noise_pred_cond.std()
            std_cfg = noise_cfg.std()
            factor = 0.7
            if std_cfg > 0:
                noise_pred_rescaled = noise_cfg * (std_cond / std_cfg)
                noise_pred = factor * noise_pred_rescaled + (1 - factor) * noise_cfg
            else:
                noise_pred = noise_cfg

            # Step
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # --- 4. Decode ---
        # Keep latents for plotting
        final_specs = latents.clone().cpu()

        # Decode Audio
        print("🔊 Decoding audio...")
        specs_batch = latents.squeeze(1)
        audio_batch = self.vocoder.decode(specs_batch)

        # --- 5. Save Files & Plot ---
        print(f"📂 Saving to {save_dir}...")
        results = {}

        for i, file_base in enumerate(file_metadata):
            # Save Audio
            wav_path = os.path.join(save_dir, file_base + ".wav")
            self.vocoder.save_audio(audio_batch[i], wav_path)

            # Save Plot (If requested)
            if plot:
                self._save_plot(final_specs[i], file_base, save_dir)

            # Group results
            inst_name = file_base.split('_')[0]
            if inst_name not in results: results[inst_name] = []
            results[inst_name].append(audio_batch[i].cpu())

        print("✨ Generation Complete!")
        return results

    def _save_plot(self, spec_tensor, title, save_dir):
        """
        Helper to plot and save the spectrogram
        """
        # spec_tensor is [1, 80, T]
        spec = spec_tensor.squeeze().numpy()

        plt.figure(figsize=(5, 3))
        plt.imshow(spec, origin='lower', cmap='inferno')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(save_dir, title + ".png"))
        plt.close()