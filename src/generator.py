import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

# Import your modules
from src.model import TimeConditionedUnet, ConcatConditionedUnet
from src.vocoder import BigVGAN_Vocoder
from src.denoising import *


class DiffusionGenerator:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        print(f"⏳ Initializing Generator on {self.device}...")

        # 1. Setup Label Map
        self.label_map = self._create_label_map(config['selected_families'])
        self.num_classes = len(self.label_map)

        # 2. Vocoder & Model
        self.vocoder = BigVGAN_Vocoder(device=self.device)

        if self.config['conditioning'] == 'time':
            self.model = TimeConditionedUnet(
                num_classes=self.num_classes + 1, num_pitches=129,
                T=config['T_target'], use_pitch=config['use_pitch']
            ).to(self.device)
        else:
            self.model = ConcatConditionedUnet(
                num_classes=self.num_classes + 1, num_pitches=129,
                T=config['T_target'], use_pitch=config['use_pitch']
            ).to(self.device)

        self._load_checkpoint()

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

    @torch.no_grad()
    @torch.no_grad()
    def generate(self, pitch=60, samples_per_class=1, output_dir=None, guidance_scale=3.0):
        """
        Optimized generation: Runs ALL instruments in a single batch (One loop over t).
        """
        save_dir = output_dir if output_dir else os.path.join(self.config['output_dir'], "generated")
        os.makedirs(save_dir, exist_ok=True)

        use_pitch = self.config.get('use_pitch', True)
        null_class = self.num_classes
        null_pitch = 128

        # --- 1. Construct the Batch ---
        # We collect all the labels we need for the entire run
        all_labels = []
        all_pitches = []
        file_metadata = []  # To keep track of which index belongs to which instrument

        for name, label_idx in self.label_map.items():
            # Add N copies for this instrument
            all_labels.extend([label_idx] * samples_per_class)
            if use_pitch:
                all_pitches.extend([pitch] * samples_per_class)

            # Record metadata for saving later
            for i in range(samples_per_class):
                file_metadata.append(f"{name}_{i + 1}.wav")

        total_samples = len(all_labels)
        print(f"🎹 Generating {total_samples} samples in parallel (Scale={guidance_scale})...")

        # Convert to Tensors
        cond_labels = torch.tensor(all_labels, device=self.device, dtype=torch.long)
        cond_pitches = torch.tensor(all_pitches, device=self.device, dtype=torch.long) if use_pitch else None

        # Create Unconditional Tensors (for CFG)
        uncond_labels = torch.full((total_samples,), null_class, device=self.device, dtype=torch.long)
        uncond_pitches = torch.full((total_samples,), null_pitch, device=self.device,
                                    dtype=torch.long) if use_pitch else None

        # --- 2. Initialize Latents ---
        latents = torch.randn(
            (total_samples, 1, 80, self.config['T_target']),
            device=self.device
        )

        # --- 3. Single Diffusion Loop ---
        for t in tqdm(self.noise_scheduler.timesteps, desc="Sampling", leave=False):
            # A. Prepare Inputs for CFG (Batch Size * 2)
            # We stack [Conditional, Unconditional]
            latent_input = torch.cat([latents] * 2)
            t_batch = torch.full((total_samples * 2,), t, device=self.device, dtype=torch.long)

            label_batch = torch.cat([cond_labels, uncond_labels])
            pitch_batch = torch.cat([cond_pitches, uncond_pitches]) if use_pitch else None

            # B. Predict Noise
            noise_pred = self.model(latent_input, t_batch, label_batch, pitch_batch)

            # C. Perform Guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # D. Rescale CFG (Fixes noise power artifacts)
            std_cond = noise_pred_cond.std()
            std_cfg = noise_cfg.std()
            factor = 0.7
            if std_cfg > 0:  # Avoid div by zero
                noise_pred_rescaled = noise_cfg * (std_cond / std_cfg)
                noise_pred = factor * noise_pred_rescaled + (1 - factor) * noise_cfg
            else:
                noise_pred = noise_cfg

            # E. Scheduler Step
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # --- 4. Decode All at Once ---
        # Input: [Total_Samples, 1, 80, T]
        specs_batch = latents.squeeze(1)

        # This calls BigVGAN once for the massive batch
        print("🔊 Decoding audio...")
        audio_batch = self.vocoder.decode(specs_batch)  # Returns [Total_Samples, 1, Audio_Len]

        # --- 5. Save Files ---
        print(f"📂 Saving to {save_dir}...")
        results = {}

        for i, filename in enumerate(file_metadata):
            path = os.path.join(save_dir, filename)
            self.vocoder.save_audio(audio_batch[i], path)

            # Optional: Group by instrument for return
            inst_name = filename.split('_')[0]
            if inst_name not in results: results[inst_name] = []
            results[inst_name].append(audio_batch[i].cpu())

        print("✨ Generation Complete!")
        return results
    
    def save_plots(self, raw_latents, clean_latents, labels, save_dir, method, instrument_names, samples_per_class):
        """
        Saves a comparison plot for each generated sample.
        """
        # We'll create one big grid plot for all generated samples
        num_samples = len(raw_latents)
        rows = 2 if method else 1

        # Limit grid size if too massive (optional, but good for safety)
        if num_samples > 20:
            print("   (Plotting first 20 samples only to save time)")
            num_samples = 20

        fig, axes = plt.subplots(rows, num_samples, figsize=(num_samples * 2.5, 3 * rows))

        # Ensure axes is 2D array [rows, cols]
        if num_samples == 1:
            axes = np.array(axes).reshape(rows, 1)
        elif rows == 1:
            axes = axes.reshape(1, -1)

        idx = 0
        for i in range(num_samples):
            # Top: Raw
            ax_raw = axes[0, i]
            raw_img = raw_latents[i].squeeze().cpu().numpy()
            ax_raw.imshow(raw_img, origin='lower', cmap='inferno')

            # Label
            inst_idx = i // samples_per_class
            inst_name = instrument_names[inst_idx]
            ax_raw.set_title(f"{inst_name} {i % samples_per_class + 1}")
            ax_raw.axis('off')

            # Bottom: Denoised (if active)
            if method:
                ax_clean = axes[1, i]
                # If using NLM, clean_latents might be float tensor from 'denoise_tensor_via_nlm'
                # If using Spectral, clean_latents is float tensor
                clean_img = clean_latents[i].squeeze().cpu().numpy()

                ax_clean.imshow(clean_img, origin='lower', cmap='inferno')
                ax_clean.set_title(f"{method}")
                ax_clean.axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, "generation_summary_plot.png")
        plt.savefig(save_path)
        print(f"   📊 Plot saved to {save_path}")
        plt.show()
        plt.close()