import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

arnold_key = 10
watermark_size = 128
max_repeat_factor = 3

def arnold_map(img, iters):
    n = img.shape[0]
    out = img.copy()
    x_idx, y_idx = np.indices((n, n))
    nx_idx = (x_idx + y_idx) % n
    ny_idx = (x_idx + 2 * y_idx) % n
    for _ in range(iters):
        temp = np.zeros_like(out)
        temp[nx_idx, ny_idx] = out[x_idx, y_idx]
        out = temp
    return out


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


def get_output_name(host_path):
    stem = Path(host_path).stem
    return f"{stem}_watermarked.png"


def main():
    parser = argparse.ArgumentParser(description="Embed watermark and save watermarked image with host-based name")
    parser.add_argument("--host", required=True, help="Host image path")
    parser.add_argument("--watermark", required=True, help="Watermark image path")
    parser.add_argument("--output-dir", default="watermarked_outputs", help="Output directory for saved files")
    args = parser.parse_args()

    if not os.path.exists(args.host):
        raise FileNotFoundError(f"Host image not found: {args.host}")
    if not os.path.exists(args.watermark):
        raise FileNotFoundError(f"Watermark image not found: {args.watermark}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    # Read color image and operate on the luminance (Y) channel to preserve color
    host_color = cv2.imread(args.host, cv2.IMREAD_COLOR)
    if host_color is None:
        raise FileNotFoundError(f"Host image could not be read: {args.host}")
    # convert to YCrCb and use Y channel for embedding (keeps embedding logic identical)
    host_ycrcb = cv2.cvtColor(host_color, cv2.COLOR_BGR2YCrCb)
    host = host_ycrcb[:, :, 0].astype(np.int32)
    h, w = host.shape

    wm = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE)
    wm = cv2.resize(wm, (watermark_size, watermark_size))
    wm_scrambled = arnold_map(wm, arnold_key)
    wm_bits = (wm_scrambled > 127).astype(np.uint8).flatten()
    # Repeat payload bits to improve robustness while keeping embedding math unchanged.
    capacity = ((h + 1) // 2) * (w // 2)
    repeat_factor = min(max_repeat_factor, max(1, capacity // wm_bits.size))
    wm_bits_embed = np.tile(wm_bits, repeat_factor)

    # Compute the original host LBP map and embed a combined bit (watermark XOR LBP) so extraction can be blind.
    host_lbp = compute_lbp_map(host.astype(np.uint8))
    watermarked = host.copy()
    location_map = []
    idx = 0

    for i in range(0, h, 2):
        for j in range(0, w - 1, 2):
            if idx >= len(wm_bits_embed):
                break

            x = int(host[i, j])
            y = int(host[i, j + 1])
            d = x - y
            a = (x + y) // 2
            wm_bit = int(wm_bits_embed[idx])
            lbp_bit = int(host_lbp[i, j] & 1)
            combined_bit = wm_bit ^ lbp_bit
            d_new = 2 * d + combined_bit
            x_new = a + (d_new + 1) // 2
            y_new = a - d_new // 2

            if 0 <= x_new <= 255 and 0 <= y_new <= 255:
                watermarked[i, j] = x_new
                watermarked[i, j + 1] = y_new
                location_map.append(1)
                idx += 1
            else:
                location_map.append(0)

        if idx >= len(wm_bits_embed):
            break

    # place modified Y back and convert to BGR before saving so color is preserved
    host_ycrcb[:, :, 0] = watermarked.astype(np.uint8)
    watermarked_color = cv2.cvtColor(host_ycrcb, cv2.COLOR_YCrCb2BGR)
    watermarked_path = output_dir / get_output_name(args.host)
    cv2.imwrite(str(watermarked_path), watermarked_color)
    np.save(output_dir / "location_map.npy", np.array(location_map, dtype=np.uint8))
    np.save(output_dir / "payload_len.npy", np.array(idx, dtype=np.int32))
    np.save(output_dir / "repeat_factor.npy", np.array(repeat_factor, dtype=np.int32))
    elapsed_time = time.perf_counter() - start_time
    np.save(output_dir / f"{Path(args.host).stem}_embed_time.npy", np.array(elapsed_time, dtype=np.float64))

    print("Embedding completed with Arnold scrambling")
    print(f"Watermarked image saved as: {watermarked_path}")
    print("Embedded bits:", idx)
    print("Repeat factor:", repeat_factor)
    print(f"Embedding time: {elapsed_time:.6f} s")


if __name__ == "__main__":
    main()

