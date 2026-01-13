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
    def generate(self, pitch=60, samples_per_class=1, output_dir=None):
        """
        Generates audio AND plots for every class.
        """
        save_dir = output_dir if output_dir else os.path.join(self.config['output_dir'], "generated")
        os.makedirs(save_dir, exist_ok=True)
        print(f"🎹 Generating {samples_per_class} samples per class...")

        # 1. Setup Batch
        instrument_names = list(self.label_map.keys())
        n_classes = len(instrument_names)
        B = n_classes * samples_per_class

        labels = torch.tensor(list(self.label_map.values()), device=self.device).long().repeat_interleave(
            samples_per_class)
        pitches = torch.full((B,), pitch, device=self.device).long() if self.config['use_pitch'] else None

        latents = torch.randn((B, 1, 80, self.config['T_target']), device=self.device)

        # 2. Diffusion Loop
        for t in tqdm(self.noise_scheduler.timesteps, desc="Generating", leave=False):
            t_batch = torch.full((B,), t, device=self.device).long()
            noise_pred = self.model(latents, t_batch, labels, pitches)
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # 3. Denoising
        raw_latents = latents.clone()  # Keep raw for plotting
        method = self.config.get('denoise_method', None)
        strength = self.config.get('denoise_strength', 0)

        if method == 'spectral':
            print(f"   🧹 Applying Spectral Gating (Strength: {strength})")
            latents = denoise_audio_tensor(latents, strength=strength)
        elif method == 'nlm':
            print(f"   🧹 Applying NLM Denoising (Strength: {strength})")
            latents = denoise_tensor_via_nlm(latents, strength=int(strength))
        elif method == 'median':
            latents = smooth_via_median(latents)
        elif method == 'gaussian':
            latents = smooth_via_gaussian(latents)
        elif method == 'custom':
            latents = smooth_via_custom_kernel(latents, kernel_list=self.config['custom_kernel'])

        # 4. Save Plots
        print("   📊 Saving Spectrogram Plots...")
        self.save_plots(raw_latents, latents, labels, save_dir, method, instrument_names, samples_per_class)

        # 5. Decode Audio
        specs_batch = latents.squeeze(1)
        audio_batch = self.vocoder.decode(specs_batch)

        # 6. Save Audio Files
        idx = 0
        results = {}
        for instrument_name in instrument_names:
            results[instrument_name] = []
            for i in range(samples_per_class):
                filename = f"{instrument_name}_{i + 1}.wav"
                self.vocoder.save_audio(audio_batch[idx], os.path.join(save_dir, filename))
                results[instrument_name].append(audio_batch[idx].cpu())
                idx += 1

        print("✨ Generation & Plotting Complete!")
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