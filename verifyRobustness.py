import argparse
from pathlib import Path

import cv2
import numpy as np


ARNOLD_KEY = 10
WATERMARK_SIZE = 128


def get_base_name(image_path):
    stem = Path(image_path).stem
    if stem.endswith("_watermarked"):
        return stem[: -len("_watermarked")]
    return stem


def inverse_arnold_map(img, iters):
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


def extract_watermark_from_attacked(attacked_y, location_map, payload_len):
    wm_img = attacked_y.astype(np.int32)
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
        padded = np.zeros(needed, dtype=np.uint8)
        padded[: bits.size] = bits
        bits = padded

    wm_scrambled = bits.reshape((WATERMARK_SIZE, WATERMARK_SIZE)) * 255
    wm_final = inverse_arnold_map(wm_scrambled.astype(np.uint8), ARNOLD_KEY)
    return (wm_final > 127).astype(np.uint8)


def compute_robustness_metrics(wm_original_bin, wm_extracted_bin):
    ref = wm_original_bin.astype(np.float64).flatten()
    ext = wm_extracted_bin.astype(np.float64).flatten()

    denom = np.sqrt(np.sum(ref * ref) * np.sum(ext * ext))
    nc = float(np.sum(ref * ext) / denom) if denom > 0 else 0.0

    ber = float(np.sum(wm_original_bin != wm_extracted_bin) / wm_original_bin.size)

    ref_std = np.std(ref)
    ext_std = np.std(ext)
    if ref_std == 0.0 or ext_std == 0.0:
        pcc = 1.0 if np.array_equal(wm_original_bin, wm_extracted_bin) else 0.0
    else:
        pcc = float(np.corrcoef(ref, ext)[0, 1])

    ed = float(np.linalg.norm(ref - ext))
    return nc, ber, pcc, ed


def main():
    parser = argparse.ArgumentParser(
        description="Verify watermark robustness (NC, BER, PCC, ED) for all attacked outputs of one input image"
    )
    parser.add_argument("--input", required=True, help="Original host image path (example: clock.png)")
    parser.add_argument("--attacked-dir", default="attacked_outputs", help="Directory containing attacked images")
    parser.add_argument("--watermark", default="watermark.png", help="Original watermark image path")
    parser.add_argument("--meta-dir", default=".", help="Directory containing location_map.npy and payload_len.npy")
    args = parser.parse_args()

    input_path = Path(args.input)
    attacked_dir = Path(args.attacked_dir)
    watermark_path = Path(args.watermark)
    meta_dir = Path(args.meta_dir)

    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return
    if not attacked_dir.exists():
        print(f"Error: Attacked directory not found: {attacked_dir}")
        return
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
    wm_original_bin = (wm_original > 127).astype(np.uint8)

    location_map = np.load(str(location_map_path))
    payload_len = int(np.load(str(payload_len_path)))

    base_name = get_base_name(input_path)
    matched_files = sorted(attacked_dir.glob(f"{base_name}_*.png"))

    if not matched_files:
        print(f"No attacked files found for '{base_name}' in: {attacked_dir}")
        return

    print(f"\nInput image: {input_path.name}")
    print(f"Attacked directory: {attacked_dir}")
    print("\nAttack                         NC         BER         PCC         ED")
    print("-" * 74)

    for attacked_file in matched_files:
        attacked_color = cv2.imread(str(attacked_file), cv2.IMREAD_COLOR)
        if attacked_color is None:
            print(f"{attacked_file.name:<28} {'READ_FAIL':>10} {'READ_FAIL':>11} {'READ_FAIL':>11} {'READ_FAIL':>10}")
            continue

        attacked_y = cv2.cvtColor(attacked_color, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        wm_extracted_bin = extract_watermark_from_attacked(attacked_y, location_map, payload_len)
        nc, ber, pcc, ed = compute_robustness_metrics(wm_original_bin, wm_extracted_bin)

        attack_name = attacked_file.stem.replace(f"{base_name}_", "", 1)
        print(f"{attack_name:<28} {nc:>10.6f} {ber:>11.6f} {pcc:>11.6f} {ed:>10.4f}")


if __name__ == "__main__":
    main()

