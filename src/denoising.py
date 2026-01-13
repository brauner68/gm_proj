import torch
import numpy as np
import cv2


def denoise_audio_tensor(spectrogram, strength=0.25, min_db=-11.5, mode='subtraction'):
    """
    Method 1: Spectral Gating with Global Floor.
    Push noise down to 'min_db' (True Silence) instead of the local noise floor.

    Args:
        spectrogram (torch.Tensor): Shape [B, 1, H, W] or [B, H, W].
        strength (float): Threshold strength. Higher = more silence.
                          Try 0.25 to 0.5.
        min_db (float): The value considered "Digital Silence" in your data.
        mode (str): 'subtraction' (Smoother, acts like a fader) or
                    'gating' (Harder, keeps signal loud, might add artifacts).
    """
    if strength <= 0:
        return spectrogram

    # 1. Standardize shape
    if spectrogram.dim() == 3:
        spectrogram = spectrogram.unsqueeze(1)

    original_shape = spectrogram.shape

    # 2. Shift to Positive Domain based on GLOBAL Minimum
    # We assume anything below min_db is silence.
    # shifted_spec range: [0.0, 13.5] (if max is 2 and min is -11.5)
    shifted_spec = spectrogram - min_db

    # Clip negatives (in case some values are below -11.5)
    shifted_spec = torch.relu(shifted_spec)

    # 3. Estimate Noise Profile
    # Calculate median across Time (dim -1)
    # noise_profile shape: [B, C, H, 1]
    noise_profile = torch.median(shifted_spec, dim=-1, keepdim=True).values

    # 4. Calculate Threshold
    threshold = noise_profile * strength

    # 5. Apply Denoising
    if mode == 'subtraction':
        # Soft Subtraction (ReLU).
        # Reduces noise volume. If (Signal < Threshold), it becomes 0 (Silence).
        # Pros: Very smooth, no clicking. Cons: Reduces volume of quiet signals.
        processed = torch.relu(shifted_spec - threshold)

    elif mode == 'gating':
        # Hard/Soft Gating Mask.
        # Preserves signal volume, only mutes the noise.
        # Create a mask (1.0 where signal > threshold, 0.0 otherwise)
        mask = (shifted_spec > threshold).float()

        # Optional: Smooth the mask slightly to reduce "clicking"
        # (Simple box blur on the time axis)
        # mask = some_smoothing_func(mask)

        processed = shifted_spec * mask

    # 6. Restore Domain
    # We add min_db back.
    # CRITICAL: Areas that were 0 (gated) now become min_db (-11.5).
    # This is TRUE SILENCE, not the old noisy floor.
    denoised = processed + min_db

    return denoised.view(original_shape)


def denoise_visual_image(spectrogram_numpy, strength=10):
    """
    Visual-only denoising for plots (unchanged).
    """
    spec_clean = np.nan_to_num(spectrogram_numpy, nan=0.0)
    img_min, img_max = spec_clean.min(), spec_clean.max()

    if (img_max - img_min) < 1e-6:
        return np.zeros_like(spec_clean, dtype=np.uint8)

    norm_img = 255 * (spec_clean - img_min) / (img_max - img_min)
    img_uint8 = norm_img.astype(np.uint8)

    return cv2.fastNlMeansDenoising(img_uint8, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def denoise_tensor_via_nlm(spectrogram_tensor, strength=10):
    """
    NLM for tensors (unchanged).
    """
    if strength <= 0: return spectrogram_tensor
    device = spectrogram_tensor.device
    spec_np = spectrogram_tensor.detach().cpu().numpy()
    denoised_batch = []

    for i in range(spec_np.shape[0]):
        img = spec_np[i].squeeze()
        img_min, img_max = img.min(), img.max()
        range_val = img_max - img_min

        if range_val < 1e-6:
            denoised_batch.append(img)
            continue

        img_uint8 = (255 * (img - img_min) / range_val).astype(np.uint8)
        clean_uint8 = cv2.fastNlMeansDenoising(img_uint8, None, h=strength, templateWindowSize=7, searchWindowSize=21)
        clean_float = (clean_uint8.astype(np.float32) / 255.0) * range_val + img_min
        denoised_batch.append(clean_float)

    result = torch.tensor(np.array(denoised_batch), device=device)
    if spectrogram_tensor.dim() == 4: result = result.unsqueeze(1)
    return result