import torch
import numpy as np
import cv2


def denoise_audio_tensor(spectrogram, strength=0.15):
    """
    Method 1: Spectral Gating (Vectorized & Range-Safe).
    Fastest method. Operates on the entire batch at once on the GPU.

    Args:
        spectrogram (torch.Tensor): Shape [B, 1, H, W] or [B, H, W].
        strength (float): Threshold factor.
    """
    if strength <= 0:
        return spectrogram

    # 1. Standardize shape to [B, 1, H, W]
    if spectrogram.dim() == 3:
        spectrogram = spectrogram.unsqueeze(1)

    original_shape = spectrogram.shape
    B, C, H, W = spectrogram.shape

    # 2. Find the 'floor' (min value) for EACH sample in the batch at once
    # We flatten H and W dimensions to find min over the spatial area
    # min_vals shape: [B, 1, 1, 1]
    min_vals = spectrogram.view(B, C, -1).min(dim=2).values.view(B, C, 1, 1)

    # 3. Shift entire batch to positive domain [0.0, ...]
    spec_shifted = spectrogram - min_vals

    # 4. Estimate Noise Profile (Median across Time Axis)
    # This runs in parallel for all B samples
    noise_profile = torch.median(spec_shifted, dim=-2, keepdim=True).values

    # 5. Calculate Threshold & Apply Gate
    # strength is relative to the noise floor we just found
    threshold = noise_profile * (1.0 + strength)

    # ReLU performs the "Gating" (clipping negatives to 0)
    spec_gated_shifted = torch.relu(spec_shifted - threshold)

    # 6. Shift Back to original range [-11.5, ...]
    denoised = spec_gated_shifted + min_vals

    return denoised.view(original_shape)


def denoise_tensor_via_nlm(spectrogram_tensor, strength=10):
    """
    Method 2: Non-Local Means (NLM).
    Cannot be vectorized because OpenCV only accepts single 2D arrays on CPU.
    """
    if strength <= 0:
        return spectrogram_tensor

    device = spectrogram_tensor.device
    spec_np = spectrogram_tensor.detach().cpu().numpy()
    denoised_batch = []

    # Loop is required here due to OpenCV limitations
    for i in range(spec_np.shape[0]):
        img = spec_np[i].squeeze()

        # Normalize to 0-255
        img_min = img.min()
        img_max = img.max()
        range_val = img_max - img_min

        if range_val < 1e-6:
            denoised_batch.append(img)
            continue

        img_uint8 = (255 * (img - img_min) / range_val).astype(np.uint8)

        # Apply OpenCV NLM
        clean_uint8 = cv2.fastNlMeansDenoising(
            img_uint8, None, h=strength, templateWindowSize=7, searchWindowSize=21
        )

        # Restore Float
        clean_float = (clean_uint8.astype(np.float32) / 255.0) * range_val + img_min
        denoised_batch.append(clean_float)

    result = torch.tensor(np.array(denoised_batch), device=device)
    if spectrogram_tensor.dim() == 4:
        result = result.unsqueeze(1)

    return result


def denoise_visual_image(spectrogram_numpy, strength=10):
    """
    Helper strictly for plotting. Returns uint8 [0-255].
    """
    spec_clean = np.nan_to_num(spectrogram_numpy, nan=0.0)
    img_min, img_max = spec_clean.min(), spec_clean.max()

    if (img_max - img_min) < 1e-6:
        return np.zeros_like(spec_clean, dtype=np.uint8)

    norm_img = 255 * (spec_clean - img_min) / (img_max - img_min)
    img_uint8 = norm_img.astype(np.uint8)

    return cv2.fastNlMeansDenoising(img_uint8, None, h=strength, templateWindowSize=7, searchWindowSize=21)