import argparse
import time
from pathlib import Path

import cv2
import numpy as np

arnold_key = 10
watermark_size = 128


def compute_lbp_map(img):
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    if h < 3 or w < 3:
        return lbp

    c = img[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    code |= ((img[:-2, :-2] >= c).astype(np.uint8) << 7)
    code |= ((img[:-2, 1:-1] >= c).astype(np.uint8) << 6)
    code |= ((img[:-2, 2:] >= c).astype(np.uint8) << 5)
    code |= ((img[1:-1, 2:] >= c).astype(np.uint8) << 4)
    code |= ((img[2:, 2:] >= c).astype(np.uint8) << 3)
    code |= ((img[2:, 1:-1] >= c).astype(np.uint8) << 2)
    code |= ((img[2:, :-2] >= c).astype(np.uint8) << 1)
    code |= ((img[1:-1, :-2] >= c).astype(np.uint8) << 0)
    lbp[1:-1, 1:-1] = code
    return lbp


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


def get_base_name(path):
    stem = Path(path).stem
    if stem.endswith("_watermarked"):
        return stem[: -len("_watermarked")]
    return stem


def main():
    parser = argparse.ArgumentParser(description="Extract watermark and restore host image")
    parser.add_argument("--watermarked", default="watermarked.png", help="Watermarked image path")
    parser.add_argument("--meta-dir", default=None, help="Directory containing location_map.npy and payload_len.npy")
    parser.add_argument("--restored-host", default="restored_host.png", help="Output restored host image path")
    parser.add_argument("--extracted-watermark", default="extracted_watermark.png", help="Output extracted watermark image path")
    args = parser.parse_args()

    wm_path = Path(args.watermarked)
    if not wm_path.exists():
        raise FileNotFoundError(f"Watermarked image not found: {wm_path}")

    meta_dir = Path(args.meta_dir) if args.meta_dir is not None else wm_path.parent
    location_map_path = meta_dir / "location_map.npy"
    payload_len_path = meta_dir / "payload_len.npy"
    if not location_map_path.exists() or not payload_len_path.exists():
        raise FileNotFoundError(f"Required metadata files not found in: {meta_dir}")

    start_time = time.perf_counter()

    # Read color image and extract luminance channel for extraction
    wm_color = cv2.imread(str(wm_path), cv2.IMREAD_COLOR)
    if wm_color is None:
        raise FileNotFoundError(f"Could not read watermarked image: {wm_path}")
    wm_ycrcb = cv2.cvtColor(wm_color, cv2.COLOR_BGR2YCrCb)
    wm_img = wm_ycrcb[:, :, 0].astype(np.int32)
    h, w = wm_img.shape

    location_map = np.load(str(location_map_path))
    payload_len = int(np.load(str(payload_len_path)))

    restored = wm_img.copy()
    extracted_bits = []
    positions = []

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
                bit = d_new & 1
                extracted_bits.append(bit)
                positions.append((i, j))
                bit_idx += 1

            d = d_new // 2
            a = (x + y) // 2

            x_orig = a + (d + 1) // 2
            y_orig = a - d // 2

            restored[i, j] = x_orig
            restored[i, j + 1] = y_orig

        if lm_idx >= len(location_map):
            break

    restored_lbp = compute_lbp_map(restored.astype(np.uint8))
    recovered_bits = []
    for bit, (i, j) in zip(extracted_bits, positions):
        lbp_bit = int(restored_lbp[i, j] & 1)
        recovered_bits.append(bit ^ lbp_bit)

    recovered_bits = np.array(recovered_bits, dtype=np.uint8)
    wm_scrambled = recovered_bits[: watermark_size * watermark_size].reshape((watermark_size, watermark_size)) * 255
    wm_final = inverse_arnold_map(wm_scrambled.astype(np.uint8), arnold_key)

    # place restored Y back into the color image and save the restored host in color
    wm_ycrcb[:, :, 0] = restored.astype(np.uint8)
    restored_color = cv2.cvtColor(wm_ycrcb, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(args.restored_host, restored_color)
    # save extracted watermark as grayscale image (same as before)
    cv2.imwrite(args.extracted_watermark, wm_final)

    elapsed_time = time.perf_counter() - start_time
    np.save(meta_dir / f"{get_base_name(wm_path)}_extraction_time.npy", np.array(elapsed_time, dtype=np.float64))

    print("Extraction completed with Arnold descrambling")
    print(f"Restored host saved as: {args.restored_host}")
    print(f"Extracted watermark saved as: {args.extracted_watermark}")
    print(f"Extraction time: {elapsed_time:.6f} s")


if __name__ == "__main__":
    main()
