import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from src.denoising import denoise_visual_image, denoise_audio_tensor

# Import your modules
from src.dataset import NSynthDataset, BigVGAN_NSynthDataset
from src.model import TimeConditionedUnet, ConcatConditionedUnet


class DiffusionTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Setup Directories
        os.makedirs(args['output_dir'], exist_ok=True)

        # 2. Prepare Data
        # We add 1 to num_classes later for the "Null/Unconditional" token
        self.dataset =  BigVGAN_NSynthDataset(
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
        # Real classes = len(dataset.label_map)
        # We add 1 extra class for the "Null" token (used for Classifier-Free Guidance)
        self.num_classes = len(self.dataset.label_map)
        self.null_class = self.num_classes  # The index of the null token
        # Pitch: MIDI 0-127. Null token = 128.
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

        # 5. Scheduler (The Diffusion Magic)
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
        print(f"Classes: {self.dataset.label_map}")
        print(f"Model will handle {self.num_classes + 1} embeddings (Index {self.null_class} is NULL)")

        # Evaluate
        self.evaluation = True
        self.plot_eval = True
        self.gen_audio_eval = False

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

                if use_pitch:
                    pitches = pitches.to(self.device)
                else:
                    pitches = None

                # --- A. Sample Noise ---
                noise = torch.randn_like(images)
                bs = images.shape[0]

                # --- B. Sample Timesteps ---
                # Random integer t for each image in batch (0 to 999)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device
                ).long()

                # --- C. Add Noise (Forward Diffusion) ---
                noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

                # --- D. CFG Logic ---
                if self.args['cfg_prob'] > 0:
                    mask = torch.rand(images.shape[0], device=self.device) < self.args['cfg_prob']
                    labels[mask] = self.null_class

                    if use_pitch:
                        pitches[mask] = self.null_pitch

                # --- E. Predict Noise (Reverse Process) ---
                noise_pred = self.model(noisy_images, timesteps, labels, pitches)

                # --- F. Loss & Backprop ---
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                progress_bar.set_description(f"Loss: {loss.item():.4f}")

                # Change lr
                if self.change_lr is not None:
                    if epoch == self.change_lr['epoch']:
                        print(f"Learning rate changed to {self.change_lr['lr']}")
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = self.change_lr['lr']
                        self.change_lr = None

            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Average Epoch Loss: {avg_loss:.4f}")
            self.loss_history.append(avg_loss)

            # Save Checkpoint
            if (epoch + 1) % self.args['save_interval'] == 0:
                self.save_checkpoint(epoch)
                if self.evaluation:
                    self.evaluate(epoch)  # Generate sample images

# ------------------------------------------------------------------------------------

    def plot_loss_curve(self):
        """Plots the training loss curve and saves it."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Training Loss')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.legend()

        # Save to the output directory
        save_path = os.path.join(self.args['output_dir'], "loss_curve.png")
        plt.savefig(save_path)
        plt.close()  # Close memory to prevent leaks
        print(f"Saved loss graph to {save_path}")

    def save_checkpoint(self, epoch):
        path = os.path.join(self.args['output_dir'], f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")

    @torch.no_grad()
    def evaluate(self, epoch):
        """
        Generates one sample for each class to visualize progress.
        """
        self.model.eval()
        print("Generating Validation Samples...")

        # We want to generate one image for every real class
        # Labels: [0, 1, 2, ..., num_classes-1]
        labels = torch.arange(self.num_classes, device=self.device)
        num_samples = len(labels)

        pitches = None
        if self.args.get('use_pitch', True):
            # Create a tensor of 60s matching the number of samples
            pitches = torch.full((num_samples,), 60, device=self.device, dtype=torch.long)

        # Start from pure random noise
        # Shape: [Num_Classes, 1, 80, T]
        latents = torch.randn(
            (num_samples, 1, 80, self.args['T_target']),
            device=self.device
        )

        # Diffusion Loop (Reverse Process)
        # We go from t=1000 down to t=0
        for t in tqdm(self.noise_scheduler.timesteps):
            # 1. Predict Noise
            # We expand t to match batch size
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(latents, t_batch, labels, pitches)

            # 2. Subtract Noise (Scheduler Step)
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Save the result as a plot
        if self.plot_eval:
            self.plot_results(latents.cpu(), labels.cpu(), epoch)

        self.model.train()  # Switch back to train mode

    def plot_results(self, images, labels, epoch):
        # Optional denoising
        if self.args['denoise_method'] == 'spectral':
            images = denoise_audio_tensor(images, strength=self.args['denoise_strength'])

        # Helper to plot grid
        num_imgs = len(images)
        fig, axes = plt.subplots(1, num_imgs, figsize=(num_imgs * 3, 3))
        if num_imgs == 1: axes = [axes]

        # Reverse map to get string names
        int_to_name = {v: k for k, v in self.dataset.label_map.items()}

        for i, ax in enumerate(axes):
            # Image is [1, 64, 64], remove channel dim
            img = images[i].squeeze().numpy()
            # If NLM, we denoise the IMAGE for visual prettiness
            if self.args['denoise_method'] == 'nlm':
                # Note: strength usually needs to be ~10-15 for NLM
                # We use the visual helper which returns uint8 directly
                img_to_plot = denoise_visual_image(img, strength=int(self.args['denoise_strength']))
            else:
                img_to_plot = img

            # Flip Y axis (Spectrograms usually have low freq at bottom)
            ax.imshow(img_to_plot, origin='lower', cmap='inferno')

            label_name = int_to_name[labels[i].item()]
            ax.set_title(f"{label_name}")
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.args['output_dir'], f"sample_epoch_{epoch+1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved validation plot to {save_path}")


