import argparse
from pathlib import Path
import cv2
import numpy as np

def compute_lbp_map(img):
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h-1):
        for j in range(1, w-1):
            c = img[i, j]
            code = 0
            code |= (img[i-1, j-1] >= c) << 7
            code |= (img[i-1, j]   >= c) << 6
            code |= (img[i-1, j+1] >= c) << 5
            code |= (img[i,   j+1] >= c) << 4
            code |= (img[i+1, j+1] >= c) << 3
            code |= (img[i+1, j]   >= c) << 2
            code |= (img[i+1, j-1] >= c) << 1
            code |= (img[i,   j-1] >= c) << 0
            lbp[i, j] = code

    return lbp

def get_base_name(image_path):
    stem = Path(image_path).stem
    if stem.endswith("_watermarked"):
        return stem[: -len("_watermarked")]
    return stem


def detect_and_save(attacked_path, lbp_ref, output_dir, output_prefix):
    img = cv2.imread(str(attacked_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image: {attacked_path}")
        return

    lbp_test = compute_lbp_map(img)
    diff = lbp_ref != lbp_test
    tampered_pixels = int(np.sum(diff))

    print(f"\nImage: {Path(attacked_path).name}")
    print("Tampered pixels:", tampered_pixels)

    if tampered_pixels == 0:
        print("Image is authentic")
        return

    print("Tampering detected")
    mask = (diff * 255).astype(np.uint8)
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[mask == 255] = [0, 0, 255]

    mask_path = output_dir / f"{output_prefix}_tamper_mask.png"
    vis_path = output_dir / f"{output_prefix}_tamper_visualization.png"
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(vis_path), overlay)
    print(f"Saved: {mask_path}")
    print(f"Saved: {vis_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect tampering using stored LBP map")
    parser.add_argument("--input", default="attacked_image.png", help="Input image path (example: clock.png)")
    parser.add_argument("--attacked-dir", default="attacked_outputs", help="Directory containing attacked images")
    parser.add_argument("--output-dir", default="tamper_outputs", help="Directory to save masks and visualizations")
    parser.add_argument("--lbp-map", default="lbp_map.npy", help="Reference LBP map (.npy)")
    args = parser.parse_args()

    lbp_map_path = Path(args.lbp_map)
    if not lbp_map_path.exists():
        print(f"Error: Reference LBP map not found: {lbp_map_path}")
        return

    lbp_ref = np.load(str(lbp_map_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    base_name = get_base_name(input_path)
    attacked_dir = Path(args.attacked_dir)
    matched_files = sorted(attacked_dir.glob(f"{base_name}_*.png")) if attacked_dir.exists() else []

    # If attacked variants are available, run batch detection for all attacks.
    if matched_files:
        print(f"Found {len(matched_files)} attacked files for '{base_name}' in: {attacked_dir}")
        for attacked_file in matched_files:
            output_prefix = attacked_file.stem
            detect_and_save(attacked_file, lbp_ref, output_dir, output_prefix)
        return

    if input_path.exists():
        detect_and_save(input_path, lbp_ref, output_dir, input_path.stem)
        return

    print(f"Error: Input image not found: {input_path}")


if __name__ == "__main__":
    main()
