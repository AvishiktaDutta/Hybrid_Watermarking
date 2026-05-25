import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import ndimage


def crop_common_region(img1, img2):
    """Return the overlapping region between two images."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return img1[:h, :w], img2[:h, :w]


def read_gray_image(path, label):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{label} not found: {path}")
    return img


def compute_psnr_ssim_nc_ber(original, compared):
    original_common, compared_common = crop_common_region(original, compared)

    original_f = original_common.astype(np.float64)
    compared_f = compared_common.astype(np.float64)

    if np.array_equal(original_common, compared_common):
        psnr_value = float("inf")
    else:
        psnr_value = peak_signal_noise_ratio(original_common, compared_common, data_range=255)
    ssim_value = structural_similarity(original_common, compared_common, data_range=255)

    denom = np.sum(original_f ** 2)
    nc_value = np.sum(original_f * compared_f) / denom if denom != 0 else 0.0
    ber_value = np.mean(original_common != compared_common)

    # compute UQI
    uqi_value = compute_uqi(original_common, compared_common)

    # compute IF (Image Fidelity)
    ref = original_common.astype(np.float64)
    diff = ref - compared_common.astype(np.float64)
    denom_if = np.sum(ref * ref)
    if denom_if == 0:
        if_value = 1.0 if np.allclose(original_common, compared_common) else 0.0
    else:
        if_value = 1.0 - (np.sum(diff * diff) / denom_if)

    # compute FSIM (approximate using gradient-magnitude similarity as a feature)
    fsim_value = compute_fsim(original_common, compared_common)

    return psnr_value, ssim_value, nc_value, ber_value, fsim_value, uqi_value, if_value


def load_scalar(path):
    if not path.exists():
        return None
    return float(np.load(str(path)))


def compute_uqi(img1, img2):
    # Universal Quality Index (Wang & Bovik)
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a2 = ((a - mu_a) ** 2).mean()
    sigma_b2 = ((b - mu_b) ** 2).mean()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    denom = (mu_a * mu_a + mu_b * mu_b) * (sigma_a2 + sigma_b2)
    if denom == 0:
        return 1.0 if np.array_equal(img1, img2) else 0.0
    return (4 * mu_a * mu_b * cov) / denom


def compute_fsim(img1, img2):
    # Approximate FSIM by using gradient-magnitude similarity as feature similarity
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    # gradient magnitude using Sobel
    gx_a = ndimage.sobel(a, axis=1, mode='reflect')
    gy_a = ndimage.sobel(a, axis=0, mode='reflect')
    gm_a = np.hypot(gx_a, gy_a)
    gx_b = ndimage.sobel(b, axis=1, mode='reflect')
    gy_b = ndimage.sobel(b, axis=0, mode='reflect')
    gm_b = np.hypot(gx_b, gy_b)

    T = 1e-6
    S_g = (2 * gm_a * gm_b + T) / (gm_a * gm_a + gm_b * gm_b + T)
    W = np.maximum(gm_a, gm_b)
    denom = np.sum(W)
    if denom == 0:
        return 1.0 if np.array_equal(img1, img2) else 0.0
    return float(np.sum(S_g * W) / denom)


def format_seconds(value):
    if value is None:
        return "N/A"
    return f"{value:.6f} s"


def resolve_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description="Verify watermark embedding and extraction results for one host image")
    parser.add_argument("--input", required=True, help="Host image path, e.g. boat.png")
    parser.add_argument("--watermark", default="watermark.png", help="Original watermark image path")
    parser.add_argument("--watermarked-dir", default="watermarked_outputs", help="Directory containing <host>_watermarked.png and payload_len.npy")
    parser.add_argument("--restored-host", default="restored_host.png", help="Restored host image path")
    parser.add_argument("--extracted-watermark", default="extracted_watermark.png", help="Extracted watermark image path")
    args = parser.parse_args()

    host_path = Path(args.input)
    if not host_path.exists():
        raise FileNotFoundError(f"Host image not found: {host_path}")

    watermark_path = Path(args.watermark)
    if not watermark_path.exists():
        raise FileNotFoundError(f"Watermark image not found: {watermark_path}")

    watermarked_dir = Path(args.watermarked_dir)
    base_name = host_path.stem
    watermarked_path = resolve_first_existing([
        watermarked_dir / f"{base_name}_watermarked.png",
        Path(f"{base_name}_watermarked.png"),
        Path("watermarked.png"),
    ])

    payload_len_path = resolve_first_existing([
        watermarked_dir / "payload_len.npy",
        Path("payload_len.npy"),
    ])

    embed_time_path = resolve_first_existing([
        watermarked_dir / f"{base_name}_embed_time.npy",
        Path(f"{base_name}_embed_time.npy"),
        Path("embed_time.npy"),
    ])

    extraction_time_path = resolve_first_existing([
        watermarked_dir / f"{base_name}_extraction_time.npy",
        Path(f"{base_name}_extraction_time.npy"),
        Path("extraction_time.npy"),
    ])

    restored_host_path = Path(args.restored_host)
    extracted_watermark_path = Path(args.extracted_watermark)

    original_host = read_gray_image(host_path, "Host image")
    watermarked_host = read_gray_image(watermarked_path, "Watermarked image")
    restored_host = read_gray_image(restored_host_path, "Restored host image") if restored_host_path.exists() else None
    original_watermark = read_gray_image(watermark_path, "Watermark image")
    extracted_watermark = read_gray_image(extracted_watermark_path, "Extracted watermark image")

    payload_len = int(np.load(str(payload_len_path)))
    bpp = payload_len / original_host.size

    host_psnr, host_ssim, host_nc, host_ber, host_fsim, host_uqi, host_if = compute_psnr_ssim_nc_ber(original_host, watermarked_host)

    watermark_resized = cv2.resize(original_watermark, (extracted_watermark.shape[1], extracted_watermark.shape[0]))
    watermark_bin = (watermark_resized > 127).astype(np.uint8) * 255
    extracted_bin = (extracted_watermark > 127).astype(np.uint8) * 255
    wm_psnr, wm_ssim, wm_nc, wm_ber, wm_fsim, wm_uqi, wm_if = compute_psnr_ssim_nc_ber(watermark_bin, extracted_bin)

    print(f"\nInput image: {host_path.name}")
    print(f"Watermarked image: {watermarked_path}")
    print(f"Extracted watermark: {extracted_watermark_path}")

    print("\nHost vs Watermarked")
    print(f"PSNR: {host_psnr}")
    print(f"SSIM: {host_ssim}")
    print(f"NC:   {host_nc}")
    print(f"BER:  {host_ber}")
    print(f"BPP:  {bpp}")

    print("\nWatermark vs Extracted Watermark")
    print(f"PSNR: {wm_psnr}")
    print(f"SSIM: {wm_ssim}")
    print(f"NC:   {wm_nc}")
    print(f"BER:  {wm_ber}")
    print(f"FSIM: {wm_fsim}")
    print(f"UQI:  {wm_uqi}")
    print(f"IF:   {wm_if}")

    if restored_host is not None:
        restored_psnr, restored_ssim, restored_nc, restored_ber, restored_fsim, restored_uqi, restored_if = compute_psnr_ssim_nc_ber(original_host, restored_host)
        print("\nHost vs Restored Host")
        print(f"PSNR: {restored_psnr}")
        print(f"SSIM: {restored_ssim}")
        print(f"NC:   {restored_nc}")
        print(f"BER:  {restored_ber}")
        print(f"FSIM: {restored_fsim}")
        print(f"UQI:  {restored_uqi}")
        print(f"IF:   {restored_if}")

    print("\nTiming")
    print(f"Embedding time:   {format_seconds(load_scalar(embed_time_path))}")
    print(f"Extraction time:  {format_seconds(load_scalar(extraction_time_path))}")

    print("\n========================================================")


if __name__ == "__main__":
    main()


