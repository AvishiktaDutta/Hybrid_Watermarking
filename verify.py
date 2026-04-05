import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def pad_to_shape(img, target_shape):
    """
    Pad cropped image back to target shape using zeros
    """
    padded = np.zeros(target_shape, dtype=img.dtype)
    h, w = img.shape
    padded[:h, :w] = img
    return padded

def crop_common_region(img1, img2):
    """
    Extract overlapping region between two images
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return img1[:h, :w], img2[:h, :w]

org = cv2.imread("clock.png", cv2.IMREAD_GRAYSCALE)
watermarked = cv2.imread("watermarked.png", cv2.IMREAD_GRAYSCALE)
restored = cv2.imread("restored_host.png", cv2.IMREAD_GRAYSCALE)
wm_t = cv2.imread("watermarked_tampered.png", cv2.IMREAD_GRAYSCALE)

wm_o = cv2.imread("watermark.png", cv2.IMREAD_GRAYSCALE)
wm_e = cv2.imread("extracted_watermark.png", cv2.IMREAD_GRAYSCALE)

if org is None or watermarked is None or restored is None or wm_t is None:
    raise FileNotFoundError("Host / Watermarked / Tampered images not found")

if wm_o is None or wm_e is None:
    raise FileNotFoundError("Watermark images not found")

wm_o = cv2.resize(wm_o, wm_e.shape[::-1])
wm_o = (wm_o > 127).astype(np.uint8)
wm_e = (wm_e > 127).astype(np.uint8)

cropped = False
if watermarked.shape != wm_t.shape:
    print("Cropping attack detected")
    wm_t = pad_to_shape(wm_t, watermarked.shape)
    cropped = True

psnr_host = peak_signal_noise_ratio(org, restored)
ssim_host = structural_similarity(org, restored)

psnr_watermark = peak_signal_noise_ratio(wm_o, wm_e)
ssim_watermark = structural_similarity(wm_o, wm_e)

wm_common, wt_common = crop_common_region(watermarked, wm_t)

nc = np.sum(wm_common * wt_common) / np.sum(wm_common ** 2)
ber = np.sum(wm_common != wt_common) / wm_common.size

payload_len = int(np.load("payload_len.npy"))
host_pixels = org.size
bpp = payload_len / host_pixels

print("\nHost vs Restored Host")
print("PSNR:", psnr_host)
print("SSIM:", ssim_host)

print("\nWatermark vs Extracted Watermark")
print("PSNR:", psnr_watermark)
print("SSIM:", ssim_watermark)

print("\nRobustness Metrics")
print("Normalized Correlation (NC):", nc)
print("Bit Error Rate (BER):", ber)

print("\nEmbedding Capacity")
print("Bits Per Pixel (BPP):", bpp)

if cropped:
    print("\nMetrics computed using padding + overlapping region due to cropping attack")

print("\n========================================================")
