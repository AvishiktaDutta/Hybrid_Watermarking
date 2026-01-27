import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# LOAD IMAGES
orig = cv2.imread("boat.png", cv2.IMREAD_GRAYSCALE)
watermarked = cv2.imread("watermarked.png", cv2.IMREAD_GRAYSCALE)
rest = cv2.imread("restored_host.png", cv2.IMREAD_GRAYSCALE)

wm_o = cv2.imread("watermark.png", cv2.IMREAD_GRAYSCALE)
wm_e = cv2.imread("extracted_watermark.png", cv2.IMREAD_GRAYSCALE)

#CHECK LOADING
if orig is None or watermarked is None or rest is None:
    raise FileNotFoundError("Host or watermarked images not found")

if wm_o is None or wm_e is None:
    raise FileNotFoundError("Watermark images not found")

#  PREPROCESS WATERMARKS
wm_o = cv2.resize(wm_o, wm_e.shape[::-1])
wm_o = (wm_o > 127).astype(np.uint8)
wm_e = (wm_e > 127).astype(np.uint8)

 # REVERSIBILITY CHECK 
""" host_identical = np.array_equal(orig, rest)
wm_identical = np.array_equal(wm_o, wm_e)

print("Host identical after extraction:", host_identical)
print("Watermark identical after extraction:", wm_identical)

# ---------------- PSNR (HOST RESTORATION) ----------------
if host_identical:
    print("Restored Host PSNR: Infinity (Perfect restoration)")
else:
    psnr_restore = peak_signal_noise_ratio(orig, rest)
    print("Restored Host PSNR:", psnr_restore) """

psnr_host = peak_signal_noise_ratio(orig, watermarked)
ssim_host = structural_similarity(orig, watermarked)

psnr_watermark = peak_signal_noise_ratio(wm_o, wm_e)
ssim_watermark = structural_similarity(wm_o, wm_e)

print("\nImperceptibility Analysis (Host vs Watermarked)")
print("PSNR:", psnr_host)
print("SSIM:", ssim_host)
print("\nImperceptibility Analysis (Watermark vs Extracted watermark)")
print("PSNR:", psnr_watermark)
print("SSIM:", ssim_watermark)
