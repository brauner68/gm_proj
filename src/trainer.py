import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json

# Import your modules
from src.dataset import NSynthDataset, BigVGAN_NSynthDataset
from src.model import TimeConditionedUnet, ConcatConditionedUnet
from src.denoising import denoise_audio_tensor, denoise_visual_image


class DiffusionTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Setup Directories
        os.makedirs(args['output_dir'], exist_ok=True)

        # Save Config
        config_path = os.path.join(args['output_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(args, f, indent=4, default=str)
            
        print(f"📄 Config saved to {config_path}")

        # 2. Prepare Data
        self.dataset = BigVGAN_NSynthDataset(
            data_path=args['data_path'],
            T_target=args['T_target'],
            max_samples=args['max_samples'],
            selected_families=args['selected_families'],
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True
        )

        # 3. Define Labels & CFG
        self.num_classes = len(self.dataset.label_map)
        self.null_class = self.num_classes
        self.num_pitches = 129
        self.null_pitch = 128

        # 4. Initialize Model
        if args['conditioning'] == 'time':
            self.model = TimeConditionedUnet(
                num_classes=self.num_classes + 1,
                num_pitches=self.num_pitches,
                T=args['T_target'],
                use_pitch=args['use_pitch']
            ).to(self.device)
        else:
            self.model = ConcatConditionedUnet(
                num_classes=self.num_classes + 1,
                num_pitches=self.num_pitches,
                T=args['T_target'],
                use_pitch=args['use_pitch']
            ).to(self.device)

        # 5. Scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=args['num_train_timesteps'],
            beta_start=args['beta_start'],
            beta_end=args['beta_end'],
            beta_schedule=args['beta_schedule'],
        )

        # 6. Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args['lr'])
        self.loss_history = []
        self.change_lr = args['change_lr']

        print(f"Trainer Initialized on {self.device}")

        # Evaluate settings
        self.evaluation = True
        self.plot_eval = True

    def train(self):
        self.model.train()
        use_pitch = self.args['use_pitch']
        script = self.args['script']

        for epoch in range(self.args['epochs']):
            print(f"Epoch {epoch + 1}/{self.args['epochs']}")
            leave = False if not script else True
            progress_bar = tqdm(self.dataloader, leave=leave)
            epoch_loss = 0

            for images, labels, pitches in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                pitches = pitches.to(self.device) if use_pitch else None

                # --- Train Step ---
                noise = torch.randn_like(images)
                bs = images.shape[0]
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,),
                                          device=self.device).long()
                noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

                # CFG
                if self.args['cfg_prob'] > 0:
                    mask = torch.rand(bs, device=self.device) < self.args['cfg_prob']
                    labels[mask] = self.null_class
                    if use_pitch: pitches[mask] = self.null_pitch

                noise_pred = self.model(noisy_images, timesteps, labels, pitches)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.loss_history.append(loss.item())
                epoch_loss += loss.item()
                progress_bar.set_description(f"Loss: {loss.item():.4f}")

                if self.change_lr and epoch == self.change_lr['epoch']:
                    for pg in self.optimizer.param_groups: pg['lr'] = self.change_lr['lr']
                    self.change_lr = None

            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Average Epoch Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.args['save_interval'] == 0:
                self.save_checkpoint(epoch)
                if self.evaluation:
                    self.evaluate(epoch)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.args['output_dir'], f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")

    def plot_loss_curve(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Training Loss')
        plt.title('Training Loss per Batch')
        plt.savefig(os.path.join(self.args['output_dir'], "loss_curve.png"))
        plt.close()

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        print("Generating Validation Samples...")

        # One sample per class
        labels = torch.arange(self.num_classes, device=self.device)
        num_samples = len(labels)
        pitches = torch.full((num_samples,), 60, device=self.device, dtype=torch.long) if self.args.get('use_pitch',
                                                                                                        True) else None

        # Start from noise
        latents = torch.randn((num_samples, 1, 80, self.args['T_target']), device=self.device)

        # Diffusion Loop
        for t in tqdm(self.noise_scheduler.timesteps, desc="Eval Loop", leave=False):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(latents, t_batch, labels, pitches)
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # --- Denoising Logic for Plotting ---
        raw_latents = latents.clone()
        denoised_latents = None

        method = self.args.get('denoise_method', None)
        strength = self.args.get('denoise_strength', 0)

        if method == 'spectral':
            # Apply audio-tensor denoising
            denoised_latents = denoise_audio_tensor(raw_latents, strength=strength)
        elif method == 'nlm':
            # For NLM, we will handle it inside the plotting loop (image level)
            # Just mark that we want to do it
            denoised_latents = raw_latents  # Placeholder, processing happens in plot_results

        if self.plot_eval:
            self.plot_results(raw_latents, denoised_latents, labels, epoch, method, strength)

        self.model.train()

    def plot_results(self, raw_images, denoised_images, labels, epoch, method, strength):
        """
        Plots Original vs Denoised (if active).
        """
        num_imgs = len(raw_images)
        # Create grid: 2 Rows (Original, Denoised) x N Columns (Classes)
        # If no denoising, just 1 Row
        rows = 2 if method else 1

        fig, axes = plt.subplots(rows, num_imgs, figsize=(num_imgs * 3, 3 * rows))

        # Handle single image case to ensure axes is always iterable properly
        if num_imgs == 1:
            axes = np.array(axes).reshape(rows, 1)  # Force 2D array
        elif rows == 1:
            axes = axes.reshape(1, -1)  # Force 2D array [1, N]

        int_to_name = {v: k for k, v in self.dataset.label_map.items()}

        for i in range(num_imgs):
            # --- 1. Plot Original (Top Row) ---
            ax_orig = axes[0, i]
            raw_img = raw_images[i].squeeze().cpu().numpy()

            ax_orig.imshow(raw_img, origin='lower', cmap='inferno')
            label_name = int_to_name[labels[i].item()]
            ax_orig.set_title(f"{label_name}")
            ax_orig.axis('off')

            # --- 2. Plot Denoised (Bottom Row) ---
            if method:
                ax_denoised = axes[1, i]
                clean_img = None

                if method == 'spectral':
                    # Already processed in tensor form
                    clean_img = denoised_images[i].squeeze().cpu().numpy()
                elif method == 'nlm':
                    # Process now on CPU image
                    clean_img = denoise_visual_image(raw_img, strength=int(strength))

                ax_denoised.imshow(clean_img, origin='lower', cmap='inferno')
                ax_denoised.set_title(f"Denoised ({method})")
                ax_denoised.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.args['output_dir'], f"sample_epoch_{epoch + 1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved validation plot to {save_path}")