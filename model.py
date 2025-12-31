import torch
from diffusers import UNet2DModel

def create_model():
    """
    Initializes the U-Net model for Class-Conditional Audio Spectrogram generation.
    Using Hugging Face's diffusers library.
    """
    model = UNet2DModel(
        # Input/Output dimensions
        sample_size=64,           # Target resolution: 64x64 Spectrograms
        in_channels=1,            # Input is 1-channel (Grayscale Log-Mel Spectrogram)
        out_channels=1,           # Output is 1-channel (Predicted Noise)
        
        # Model Depth & Width
        layers_per_block=2,       # Number of ResNet layers per down/up block (Standard depth)
        block_out_channels=(128, 128, 256, 256), # Channel width increases as resolution decreases
        
        # Downsampling Blocks (Encoder)
        down_block_types=(
            "DownBlock2D",        # High resolution: Standard convolution
            "DownBlock2D",        # High resolution: Standard convolution
            "AttnDownBlock2D",    # Low resolution: Convolution + Self-Attention (Global context)
            "DownBlock2D",        # Bottom resolution
        ),
        
        # Upsampling Blocks (Decoder) - Mirroring the Encoder
        up_block_types=(
            "UpBlock2D",          
            "AttnUpBlock2D",      # Restore resolution + Self-Attention
            "UpBlock2D",          
            "UpBlock2D",          
        ),
        
        # Conditioning Mechanism
        # "identity" allows us to inject custom embeddings (Instrument + Pitch) 
        # directly into the model's time-embedding layer.
        class_embed_type="identity" 
    )
    
    return model

if __name__ == "__main__":
    # Quick sanity check to verify model architecture
    model = create_model()
    print(f"Model initialized successfully.")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")