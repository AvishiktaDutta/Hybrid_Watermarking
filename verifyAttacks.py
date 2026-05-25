import argparse
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ARNOLD_KEY = 10
WATERMARK_SIZE = 128

def get_base_name(image_path):
    """Extract filename stem, removing _watermarked suffix if present."""
    stem = Path(image_path).stem
    if stem.endswith("_watermarked"):
        return stem[: -len("_watermarked")]
    return stem


def crop_common_region(img1, img2):
    """Return overlapping region from both images."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return img1[:h, :w], img2[:h, :w]


def inverse_arnold_map(img, iters):
    """Inverse Arnold transform used in extraction."""
    n = img.shape[0]
    out = img.copy()
    for _ in range(iters):
        temp = np.zeros_like(out)
        for x in range(n):
            for y in range(n):
                nx = (2 * x - y) % n
                ny = (-x + y) % n
                temp[nx, ny] = out[x, y]
        out = temp
    return out


def compute_metrics(original_gray, attacked_gray):
    """Compute PSNR, SSIM and BER on a common region."""
    org_common, atk_common = crop_common_region(original_gray, attacked_gray)

    psnr_value = peak_signal_noise_ratio(org_common, atk_common)
    ssim_value = structural_similarity(org_common, atk_common)
    ber_value = np.sum(org_common != atk_common) / org_common.size

    return psnr_value, ssim_value, ber_value


def extract_watermark_from_attacked(attacked_gray, location_map, payload_len):
    """Extract scrambled watermark bits from attacked image and descramble."""
    wm_img = attacked_gray.astype(np.int32)
    h, w = wm_img.shape
    bits = []

    lm_idx = 0
    bit_idx = 0

    for i in range(0, h, 2):
        for j in range(0, w - 1, 2):
            if lm_idx >= len(location_map):
                break

            flag = location_map[lm_idx]
            lm_idx += 1

            if flag == 0:
                continue

            x = int(wm_img[i, j])
            y = int(wm_img[i, j + 1])
            d_new = x - y

            if bit_idx < payload_len:
                bits.append(d_new & 1)
                bit_idx += 1

        if lm_idx >= len(location_map) or bit_idx >= payload_len:
            break

    needed = WATERMARK_SIZE * WATERMARK_SIZE
    bits = np.array(bits[:needed], dtype=np.uint8)
    if bits.size < needed:
        padded_bits = np.zeros(needed, dtype=np.uint8)
        padded_bits[: bits.size] = bits
        bits = padded_bits

    wm_scrambled = bits.reshape((WATERMARK_SIZE, WATERMARK_SIZE)) * 255
    wm_final = inverse_arnold_map(wm_scrambled.astype(np.uint8), ARNOLD_KEY)
    return wm_final


def main():
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM and BER for all attacked variants of an input image"
    )
    parser.add_argument("--input", required=True, help="Original input image path (example: boat.png)")
    parser.add_argument(
        "--attacked-dir",
        default="attacked_outputs",
        help="Directory containing attacked images",
    )
    parser.add_argument(
        "--watermark",
        default="watermark.png",
        help="Original watermark image path",
    )
    parser.add_argument(
        "--meta-dir",
        default=".",
        help="Directory containing location_map.npy and payload_len.npy",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    attacked_dir = Path(args.attacked_dir)
    watermark_path = Path(args.watermark)
    meta_dir = Path(args.meta_dir)

    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return

    if not attacked_dir.exists():
        print(f"Error: Attacked output directory not found: {attacked_dir}")
        return

    # Read color and use luminance (Y) channel for comparisons so color is preserved elsewhere
    original_color = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if original_color is None:
        print(f"Error: Could not read input image: {input_path}")
        return
    original_ycrcb = cv2.cvtColor(original_color, cv2.COLOR_BGR2YCrCb)
    original = original_ycrcb[:, :, 0]

    if not watermark_path.exists():
        print(f"Error: Watermark image not found: {watermark_path}")
        return

    location_map_path = meta_dir / "location_map.npy"
    payload_len_path = meta_dir / "payload_len.npy"

    if not location_map_path.exists() or not payload_len_path.exists():
        print(f"Error: Metadata files not found in: {meta_dir}")
        print("Required: location_map.npy and payload_len.npy")
        return

    wm_original = cv2.imread(str(watermark_path), cv2.IMREAD_GRAYSCALE)
    if wm_original is None:
        print(f"Error: Could not read watermark image: {watermark_path}")
        return

    wm_original = cv2.resize(wm_original, (WATERMARK_SIZE, WATERMARK_SIZE))
    wm_original = (wm_original > 127).astype(np.uint8)

    location_map = np.load(str(location_map_path))
    payload_len = int(np.load(str(payload_len_path)) )

    base_name = get_base_name(input_path)
    matched_files = sorted(attacked_dir.glob(f"{base_name}_*.png"))

    if not matched_files:
        print(f"No attacked files found for '{base_name}' in: {attacked_dir}")
        return

    print(f"\nInput image: {input_path.name}")
    print(f"Attacked directory: {attacked_dir}")
    print("\nAttack                     C-PSNR       C-SSIM       C-BER       W-PSNR       W-SSIM       W-BER")
    print("-" * 106)

    for attacked_file in matched_files:
        attacked_color = cv2.imread(str(attacked_file), cv2.IMREAD_COLOR)
        if attacked_color is None:
            print(
                f"{attacked_file.name:<26} {'READ_FAIL':>10} {'READ_FAIL':>12} {'READ_FAIL':>11}"
                f" {'READ_FAIL':>12} {'READ_FAIL':>12} {'READ_FAIL':>11}"
            )
            continue
        attacked_ycrcb = cv2.cvtColor(attacked_color, cv2.COLOR_BGR2YCrCb)
        attacked = attacked_ycrcb[:, :, 0]

        c_psnr, c_ssim, c_ber = compute_metrics(original, attacked)
        wm_extracted = extract_watermark_from_attacked(attacked, location_map, payload_len)
        wm_extracted_bin = (wm_extracted > 127).astype(np.uint8)
        w_psnr, w_ssim, w_ber = compute_metrics(wm_original, wm_extracted_bin)
        attack_name = attacked_file.stem.replace(f"{base_name}_", "", 1)
        print(
            f"{attack_name:<26} {c_psnr:>10.4f} {c_ssim:>12.6f} {c_ber:>11.6f}"
            f" {w_psnr:>12.4f} {w_ssim:>12.6f} {w_ber:>11.6f}"
        )


if __name__ == "__main__":
    main()

