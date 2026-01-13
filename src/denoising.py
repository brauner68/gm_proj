import torch
import numpy as np
import cv2
import torch.nn.functional as F


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
    noise_profile = torch.median(spec_shifted, dim=-1, keepdim=True).values

    # 5. Calculate Threshold & Apply Gate
    # strength is relative to the noise floor we just found
    threshold = noise_profile * strength

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


def smooth_via_median(spectrogram, kernel_size=3):
    """
    Option 2: Horizontal Median Filter.
    - Excellent for removing "salt & pepper" noise (random pixel dots)
    - Preserves edges better than Gaussian (doesn't blur the attack as much).

    Args:
        kernel_size (int): Size of the window (must be odd, e.g., 3, 5).
    """
    if kernel_size <= 1:
        return spectrogram

    # Ensure odd kernel size
    if kernel_size % 2 == 0: kernel_size += 1
    padding = kernel_size // 2

    if spectrogram.dim() == 3:
        spectrogram = spectrogram.unsqueeze(1)

    # Pad the Time axis (last dim) to maintain size
    # Pad format: (pad_left, pad_right, pad_top, pad_bottom, ...)
    spec_padded = F.pad(spectrogram, (padding, padding, 0, 0), mode='reflect')

    # Unfold creates sliding windows.
    # Input: [B, C, H, W_padded] -> Unfold Time axis
    # Output: [B, C, H, W_original, kernel_size]
    unfolded = spec_padded.unfold(dimension=-1, size=kernel_size, step=1)

    # Compute Median over the window dimension
    # Note: torch.median returns (values, indices) tuple
    smoothed = unfolded.median(dim=-1).values

    return smoothed


def smooth_via_gaussian(spectrogram, kernel_size=3, sigma=1.0):
    """
    Option 3: Horizontal Gaussian Blur (1D Convolution).
    - Best for general "softness" and connecting broken harmonic lines.
    - Will slightly blur the attack (start) of notes.
    """
    if kernel_size <= 1:
        return spectrogram

    if kernel_size % 2 == 0: kernel_size += 1
    padding = kernel_size // 2

    # 1. Create 1D Gaussian Kernel
    x = torch.arange(kernel_size, dtype=spectrogram.dtype, device=spectrogram.device) - padding
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # Normalize so sum = 1

    # Reshape for conv1d: [Out_Channels, In_Channels/Groups, Kernel_Size]
    # We want to apply same kernel to every channel independently -> Depthwise Conv
    # Shape: [Channels, 1, Kernel_Size] (where Channels = H * C usually, but here we treat H as batch or channels)

    # Trick: Treat Height (H) as "Batch" or "Channels" to simplify 1D conv over Time (W)
    B, C, H, W = spectrogram.shape

    # Reshape to [B * C * H, 1, W] to apply 1D conv per row independently
    spec_reshaped = spectrogram.view(-1, 1, W)

    # Kernel shape: [1, 1, K]
    kernel = kernel.view(1, 1, -1)

    # Pad
    spec_padded = F.pad(spec_reshaped, (padding, padding), mode='reflect')

    # Apply Conv1d
    smoothed = F.conv1d(spec_padded, kernel)

    return smoothed.view(B, C, H, W)


def smooth_via_custom_kernel(spectrogram, kernel_list):
    """
    Applies 1D Convolution along the Time Axis using a custom user-defined kernel.

    Args:
        spectrogram (torch.Tensor): Shape [B, 1, H, W].
        kernel_list (list): A list of floats, e.g., [0, 1, 0] or [0.1, 0.8, 0.1].
                            Must be odd length.
    """
    if not kernel_list:
        return spectrogram

    # 1. Prepare Kernel
    device = spectrogram.device
    k_tensor = torch.tensor(kernel_list, dtype=spectrogram.dtype, device=device)

    # Normalize kernel so it doesn't change volume (sum = 1.0)
    if k_tensor.sum() != 0:
        k_tensor = k_tensor / k_tensor.sum()

    k_size = len(k_tensor)
    if k_size % 2 == 0:
        raise ValueError("Custom kernel size must be ODD (e.g., 3, 5).")

    padding = k_size // 2

    # 2. Reshape for Vectorized 1D Conv
    # We treat every frequency row (H) in every batch item (B) as an independent line.
    if spectrogram.dim() == 3:
        spectrogram = spectrogram.unsqueeze(1)

    B, C, H, W = spectrogram.shape

    # Flatten Batch, Channel, and Freq into one massive "Batch" dim
    # Shape: [B*C*H, 1, W]
    spec_reshaped = spectrogram.view(-1, 1, W)

    # Reshape Kernel: [Out_Channels, In_Channels, Width] -> [1, 1, K]
    kernel_reshaped = k_tensor.view(1, 1, -1)

    # 3. Apply Padding & Convolution
    # Pad left/right with reflection to handle edges smoothly
    spec_padded = F.pad(spec_reshaped, (padding, padding), mode='reflect')

    smoothed = F.conv1d(spec_padded, kernel_reshaped)

    # 4. Restore Shape
    return smoothed.view(B, C, H, W)


def smooth_via_median(spectrogram, kernel_size=3):
    """ Horizontal Median Filter (Good for 'salt & pepper' noise). """
    if kernel_size <= 1: return spectrogram
    if kernel_size % 2 == 0: kernel_size += 1
    padding = kernel_size // 2

    if spectrogram.dim() == 3: spectrogram = spectrogram.unsqueeze(1)

    spec_padded = F.pad(spectrogram, (padding, padding, 0, 0), mode='reflect')
    unfolded = spec_padded.unfold(dimension=-1, size=kernel_size, step=1)
    return unfolded.median(dim=-1).values


# =============================================================================
# 3. VISUAL / OTHER
# =============================================================================

def denoise_visual_image(spectrogram_numpy, strength=10):
    """ Strictly for plotting (uint8 NLM). """
    spec_clean = np.nan_to_num(spectrogram_numpy, nan=0.0)
    img_min, img_max = spec_clean.min(), spec_clean.max()
    if (img_max - img_min) < 1e-6: return np.zeros_like(spec_clean, dtype=np.uint8)
    norm_img = 255 * (spec_clean - img_min) / (img_max - img_min)
    img_uint8 = norm_img.astype(np.uint8)
    return cv2.fastNlMeansDenoising(img_uint8, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def denoise_tensor_via_nlm(spectrogram_tensor, strength=10):
    """ NLM wrapper for Tensors (Slow, CPU-based). """
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

