import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ConcatConditionedUnet(nn.Module):
    def __init__(self, num_classes, num_pitches=129, T=160, class_emb_size=32, use_pitch=True):
        """
        Args:
            num_classes (int): Total instrument classes (+1 for null).
            num_pitches (int): Total pitch classes (+1 for null).
            T (int): Time dimension size.
            class_emb_size (int): Size of embedding vector for both class and pitch.
            use_pitch (bool): Whether to use pitch conditioning.
        """
        super().__init__()
        self.use_pitch = use_pitch
        self.class_emb_size = class_emb_size

        # 1. Embeddings
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        if self.use_pitch:
            self.pitch_emb = nn.Embedding(num_pitches, class_emb_size)

        # 2. Calculate Input Channels
        # Base: 1 (The noisy audio spectrogram)
        # + Class Embedding Channels
        # + Pitch Embedding Channels (if enabled)
        input_channels = 1 + class_emb_size
        if self.use_pitch:
            input_channels += class_emb_size

        # 3. The Core UNet Model
        self.model = UNet2DModel(
            sample_size=(80, T),
            in_channels=input_channels,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"
            ),
            up_block_types=(
                "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"
            ),
        )

    def forward(self, x, t, class_labels, pitch_labels=None):
        """
        x: [Batch, 1, 80, T]
        class_labels: [Batch]
        pitch_labels: [Batch] (Optional)
        """
        bs, ch, w, h = x.shape

        # --- A. Class Conditioning ---
        # 1. Look up embedding -> [Batch, emb_size]
        class_cond = self.class_emb(class_labels)
        # 2. Expand to image size -> [Batch, emb_size, 80, T]
        class_cond = class_cond.view(bs, -1, 1, 1).expand(bs, -1, w, h)

        # --- B. Pitch Conditioning ---
        if self.use_pitch:
            if pitch_labels is None:
                raise ValueError("Model is expecting pitch_labels, but None was passed.")

            # 1. Look up embedding
            pitch_cond = self.pitch_emb(pitch_labels)
            # 2. Expand to image size
            pitch_cond = pitch_cond.view(bs, -1, 1, 1).expand(bs, -1, w, h)

            # --- C. Concatenate All ---
            # Order: [Image, Class, Pitch]
            net_input = torch.cat((x, class_cond, pitch_cond), 1)
        else:
            # Order: [Image, Class]
            net_input = torch.cat((x, class_cond), 1)

        # --- D. Forward Pass ---
        return self.model(net_input, t).sample


class TimeConditionedUnet(nn.Module):
    def __init__(self, num_classes, num_pitches=129, T=160, use_pitch=True):
        super().__init__()
        self.use_pitch = use_pitch

        # Calculate total unique combinations
        # If use_pitch is False, we just use num_classes
        # If use_pitch is True, we have (Classes * Pitches) combinations
        if self.use_pitch:
            self.total_embeddings = num_classes * num_pitches
        else:
            self.total_embeddings = num_classes

        # We let diffusers handle the embedding internally
        self.model = UNet2DModel(
            sample_size=(80, T),
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),

            # PASS THE TOTAL COUNT HERE
            num_class_embeds=self.total_embeddings
        )

    def forward(self, x, t, class_labels, pitch_labels=None):
        # 1. Flatten the labels into a single integer
        # New ID = (Class_ID * 129) + Pitch_ID
        if self.use_pitch:
            if pitch_labels is None:
                # Fallback: Treat as pitch 0 or handle error
                pitch_labels = torch.zeros_like(class_labels)

            # Combine them
            combined_labels = (class_labels * 129) + pitch_labels
        else:
            combined_labels = class_labels

        # 2. Pass standard integer labels to the model
        return self.model(x, t, class_labels=combined_labels).sample