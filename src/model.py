import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ConcatConditionedUnet(nn.Module):
    def __init__(self, num_classes, class_emb_size=32):
        """
        Args:
            num_classes (int): Total number of instrument classes (plus 1 for null/unconditional if you use CFG).
            class_emb_size (int): Size of the embedding vector (e.g., 32 or 64).
        """
        super().__init__()

        # 1. The Embedding Layer
        # Maps integer labels (0, 1, 2...) to a vector (e.g., size 32)
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # 2. The Core UNet Model
        # We use a deeper configuration than the example to handle 64x64 audio details.
        self.model = UNet2DModel(
            sample_size=64,  # Target: 64x64 spectrograms
            in_channels=1 + class_emb_size,  # 1 (Audio) + Embedding Size
            out_channels=1,  # Output: 1 Channel (Spectrogram)
            layers_per_block=2,  # 2 ResNet layers per block (Standard)
            block_out_channels=(64, 128, 128, 256),  # Deeper channel progression
            down_block_types=(
                "DownBlock2D",  # 64 -> 32
                "DownBlock2D",  # 32 -> 16
                "AttnDownBlock2D",  # 16 -> 8 (Attention is good here)
                "AttnDownBlock2D",  # 8 -> 4
            ),
            up_block_types=(
                "AttnUpBlock2D",  # 4 -> 8
                "AttnUpBlock2D",  # 8 -> 16
                "UpBlock2D",  # 16 -> 32
                "UpBlock2D",  # 32 -> 64
            ),
        )

    def forward(self, x, t, class_labels):
        """
        x: Noisy images [Batch, 1, 64, 64]
        t: Timesteps [Batch]
        class_labels: Class IDs [Batch]
        """
        # 1. Get dimensions
        bs, ch, w, h = x.shape

        # 2. Create Class Conditioning
        # Look up embedding -> [Batch, class_emb_size]
        class_cond = self.class_emb(class_labels)

        # Reshape to image dimensions -> [Batch, class_emb_size, 1, 1]
        class_cond = class_cond.view(bs, -1, 1, 1)

        # Expand to match image size -> [Batch, class_emb_size, 64, 64]
        class_cond = class_cond.expand(bs, -1, w, h)

        # 3. Concatenate Input
        # We stack the noisy image and the class map together
        # Result shape: [Batch, 1 + class_emb_size, 64, 64]
        net_input = torch.cat((x, class_cond), 1)

        # 4. Forward Pass
        # .sample is required because diffusers returns a wrapper object
        return self.model(net_input, t).sample


class TimeConditionedUnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Note: No manual Embedding layer needed!
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            num_class_embeds=num_classes,
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, x, t, class_labels):
        return self.model(x, t, class_labels=class_labels).sample