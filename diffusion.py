import torch
from diffusers import DDPMScheduler
from tqdm import tqdm # Library for progress bars

def get_scheduler():
    """
    Creates the Noise Scheduler (The mathematician of the project).
    It defines how noise is added (Training) and removed (Inference).
    """
    # We use the standard DDPM Scheduler as requested in the brief.
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,   # T = 1000 steps
        beta_start=0.0001,          # Starting noise level (Linear Schedule)
        beta_end=0.02,              # Ending noise level
        beta_schedule="linear",     # Linearly increasing noise
        prediction_type="epsilon"   # The model predicts the Noise (Epsilon), not the image
    )
    return scheduler

@torch.no_grad() # Disable gradient calculation to save memory during generation
def sample(model, scheduler, batch_size, label_embeddings, device="cpu"):
    """
    The Sampling Loop (Inference).
    Generates clean spectrograms from pure noise, guided by the label_embeddings.
    
    Args:
        model: The trained U-Net.
        scheduler: The DDPMScheduler.
        batch_size: How many samples to generate at once.
        label_embeddings: Tensor containing the combined Instrument+Pitch vectors.
        device: 'cuda' or 'cpu'.
    """
    
    # 1. Prepare the canvas: Start with pure Gaussian Noise (x_T)
    # Shape: (Batch_Size, Channels=1, Height=64, Width=64)
    image_shape = (batch_size, 1, 64, 64)
    image = torch.randn(image_shape).to(device)
    
    # 2. Set the scheduler to inference mode
    scheduler.set_timesteps(1000) # Use all 1000 steps for maximum quality
    
    print(f"Generating {batch_size} samples...")
    
    # 3. The Reverse Process Loop (T -> 0)
    # We iterate backwards from step 1000 down to 0.
    for t in tqdm(scheduler.timesteps):
        
        # 3.1 Predict the Noise
        # We ask the model: "Given this noisy image 'image' at time 't', 
        # and knowing we want 'label_embeddings' - what is the noise?"
        model_output = model(
            image, 
            t, 
            class_labels=label_embeddings # Injecting the condition (Instrument/Pitch)
        ).sample
        
        # 3.2 Remove the Noise (The Step)
        # The scheduler calculates the previous step (x_{t-1}) mathematically.
        # It subtracts the predicted noise carefully.
        image = scheduler.step(
            model_output, 
            t, 
            image
        ).prev_sample
        
    # 4. Return the clean spectrograms
    return image