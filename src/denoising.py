import torch
import numpy as np
import cv2


def denoise_audio_tensor(spectrogram, strength=1.0):
    """
    Method 1: Spectral Gating (Spectral Subtraction).
    - Fast, keeps data as Floats (High Fidelity).
    - Best for: General hiss reduction, subtle cleaning.
    """
    if strength <= 0:
        return spectrogram

    original_shape = spectrogram.shape
    # Ensure [B, 1, H, W] for consistency
    if spectrogram.dim() == 3:
        spectrogram = spectrogram.unsqueeze(1)

    # Estimate Noise Profile (Median across Time)
    noise_profile = torch.median(spectrogram, dim=-1, keepdim=True).values
    threshold = noise_profile * strength

    # Soft Subtraction
    denoised = torch.relu(spectrogram - threshold)

    return denoised.view(original_shape)


def denoise_tensor_via_nlm(spectrogram_tensor, strength=10):
    """
    Method 2: Non-Local Means (NLM) wrapper for Tensors.
    - Slower, converts to Int8 (Low Fidelity) and back.
    - Best for: Heavy restoration where structure matters more than texture.
    """
    if strength <= 0:
        return spectrogram_tensor

    device = spectrogram_tensor.device
    # Convert to Numpy (CPU)
    spec_np = spectrogram_tensor.detach().cpu().numpy()

    # We process each item in the batch individually
    denoised_batch = []

    for i in range(spec_np.shape[0]):
        # Get single spectrogram (removing channel dim if present)
        img = spec_np[i].squeeze()

        # --- 1. Normalize to 0-255 (Save min/max to restore later) ---
        img_min = img.min()
        img_max = img.max()
        range_val = img_max - img_min + 1e-6

        # Map to 0-255
        img_uint8 = (255 * (img - img_min) / range_val).astype(np.uint8)

        # --- 2. Apply OpenCV NLM ---
        # h=strength
        clean_uint8 = cv2.fastNlMeansDenoising(
            img_uint8, None, h=strength, templateWindowSize=7, searchWindowSize=21
        )

        # --- 3. Restore to Float Domain ---
        clean_float = (clean_uint8.astype(np.float32) / 255.0) * range_val + img_min
        denoised_batch.append(clean_float)

    # Stack back into a tensor
    result = torch.tensor(np.array(denoised_batch), device=device)

    # Restore original dimensions [B, 1, H, W]
    if spectrogram_tensor.dim() == 4:
        result = result.unsqueeze(1)

    return result


def denoise_visual_image(spectrogram_numpy, strength=10):
    """
    Simple helper strictly for plotting (returns uint8).
    """
    spec_clean = np.nan_to_num(spectrogram_numpy, nan=0.0)
    img_min, img_max = spec_clean.min(), spec_clean.max()

    norm_img = 255 * (spec_clean - img_min) / (img_max - img_min + 1e-6)
    img_uint8 = norm_img.astype(np.uint8)

    return cv2.fastNlMeansDenoising(img_uint8, None, h=strength, templateWindowSize=7, searchWindowSize=21)