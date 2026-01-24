from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

# Load the original and extracted watermarks as grayscale
original_img = cv2.imread("watermark.png", cv2.IMREAD_GRAYSCALE)
extracted_img = cv2.imread("extracted_watermark.png", cv2.IMREAD_GRAYSCALE)

# Resize original to 128x128 if needed (same as in embed.py)
if original_img.shape != (128, 128):
    original_img = cv2.resize(original_img, (128, 128))

# Threshold both to binary (0/1) for fair comparison (same threshold as embed.py)
original_bin = cv2.threshold(original_img, 127, 1, cv2.THRESH_BINARY)[1]
extracted_bin = cv2.threshold(extracted_img, 127, 1, cv2.THRESH_BINARY)[1]

# Convert to uint8 for PSNR calculation (0-255 range)
original_bin_255 = original_bin.astype(np.uint8) * 255
extracted_bin_255 = extracted_bin.astype(np.uint8) * 255

# Calculate MSE
mse = np.mean((original_bin_255.astype(np.float64) - extracted_bin_255.astype(np.float64)) ** 2)

# Calculate PSNR
if mse == 0:
    psnr_score = float('inf')
    print(f"PSNR Score: Infinity dB (MSE = 0)")
else:
    psnr_score = psnr(original_bin_255, extracted_bin_255)
    print(f"PSNR Score: {psnr_score:.4f} dB")
    print(f"MSE: {mse:.10f}")

# Calculate SSIM
ssim_score = ssim(original_bin_255, extracted_bin_255)
print(f"SSIM Score: {ssim_score:.4f}")

# Verify perfect match
if np.array_equal(original_bin, extracted_bin):
    print("Perfect match: All pixels are identical!")
else:
    diff_count = np.sum(original_bin != extracted_bin)
    print(f"Warning: {diff_count} pixels differ")