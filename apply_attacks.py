import argparse
import os
from pathlib import Path
import cv2
import numpy as np


def ensure_output_dir(output_dir="attacked_outputs"):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def get_base_name(image_path):
    """Extract filename without extension, removing _watermarked suffix if present"""
    stem = Path(image_path).stem
    if stem.endswith("_watermarked"):
        return stem[: -len("_watermarked")]
    return stem


def apply_jpeg_compression(image_path, output_path, quality_level=5):
    """Apply JPEG compression attack"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality_level)]
    ok, encoded = cv2.imencode(".jpg", img, encode_param)
    if not ok:
        raise IOError("Failed to apply JPEG encoding")

    degraded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if degraded is None:
        raise IOError("Failed to decode JPEG-compressed image")

    # A second JPEG cycle increases visible compression artifacts.
    ok2, encoded2 = cv2.imencode(".jpg", degraded, encode_param)
    if not ok2:
        raise IOError("Failed to apply second JPEG encoding")

    degraded = cv2.imdecode(encoded2, cv2.IMREAD_UNCHANGED)
    if degraded is None:
        raise IOError("Failed to decode second JPEG-compressed image")

    if not cv2.imwrite(output_path, degraded):
        raise IOError(f"Failed to save JPEG-attacked image to {output_path}")

    return output_path


def apply_copy_move_forgery(image_path, output_path):
    """Apply copy-move forgery attack"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]
    block_h = max(32, h // 4)
    block_w = max(32, w // 4)

    # Copy a large block from bottom-right and paste near top-left where
    # embedding starts, so the modification affects embedded pairs.
    y1 = max(0, h - block_h - 1)
    x1 = max(0, w - block_w - 1)
    y2 = min(y1 + block_h, h)
    x2 = min(x1 + block_w, w)

    source_block = img[y1:y2, x1:x2].copy()

    ty = min(max(0, h // 8), max(0, h - (y2 - y1)))
    tx = min(max(0, w // 8), max(0, w - (x2 - x1)))
    img[ty:ty + (y2 - y1), tx:tx + (x2 - x1)] = source_block

    if not cv2.imwrite(output_path, img):
        raise IOError(f"Failed to save forgery image to {output_path}")

    return output_path


def apply_salt_and_pepper(image_path, output_path, amount=0.05):
    """Apply salt-and-pepper noise attack"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    output = img.copy()
    total = img.shape[0] * img.shape[1]
    num_salt = int(np.ceil(amount * total * 0.5))
    num_pepper = int(np.ceil(amount * total * 0.5))

    for _ in range(num_salt):
        y = np.random.randint(0, img.shape[0])
        x = np.random.randint(0, img.shape[1])
        if img.ndim == 2:
            output[y, x] = 255
        else:
            output[y, x, :] = 255

    for _ in range(num_pepper):
        y = np.random.randint(0, img.shape[0])
        x = np.random.randint(0, img.shape[1])
        if img.ndim == 2:
            output[y, x] = 0
        else:
            output[y, x, :] = 0

    if not cv2.imwrite(output_path, output):
        raise IOError(f"Failed to save salt-and-pepper image to {output_path}")
    
    return output_path


def apply_gaussian_blur(image_path, output_path, kernel_size=5):
    """Apply Gaussian blur attack"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    if not cv2.imwrite(output_path, blurred):
        raise IOError(f"Failed to save blurred image to {output_path}")
    
    return output_path


def apply_rotation(image_path, output_path, angle=5):
    """Apply rotation attack"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    
    if not cv2.imwrite(output_path, rotated):
        raise IOError(f"Failed to save rotated image to {output_path}")
    
    return output_path


def apply_scaling(image_path, output_path, scale=0.8):
    """Apply scaling attack"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad back to original size
    padded = np.zeros((h, w, 3 if img.ndim == 3 else 1), dtype=img.dtype)
    padded[:new_h, :new_w] = scaled
    
    if not cv2.imwrite(output_path, padded):
        raise IOError(f"Failed to save scaled image to {output_path}")
    
    return output_path


def apply_histogram_equalization(image_path, output_path):
    """Apply histogram equalization attack"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if len(img.shape) == 3:
        # For color images, convert to HSV, equalize V channel, and convert back
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        equalized = cv2.equalizeHist(img)
    
    if not cv2.imwrite(output_path, equalized):
        raise IOError(f"Failed to save equalized image to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Apply various attacks to an input image and save results with input name as prefix")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--base-name", default=None, help="Base name for output files (defaults to input filename)")
    parser.add_argument("--output-dir", default="attacked_outputs", help="Output directory for attacked images")
    parser.add_argument("--jpeg-quality", type=int, default=5, help="JPEG compression quality (0-100)")
    parser.add_argument("--sp-amount", type=float, default=0.05, help="Salt-and-pepper noise amount (0-1)")
    parser.add_argument("--blur-kernel", type=int, default=5, help="Gaussian blur kernel size (odd number)")
    parser.add_argument("--rotation-angle", type=float, default=5, help="Rotation angle in degrees")
    parser.add_argument("--scale-factor", type=float, default=0.8, help="Scaling factor (0-1)")
    
    args = parser.parse_args()

    # Verify input exists
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        return

    output_dir = ensure_output_dir(args.output_dir)
    base_name = args.base_name if args.base_name else get_base_name(args.input)

    attacks = [
        ("jpegCompressed", lambda out: apply_jpeg_compression(args.input, out, args.jpeg_quality)),
        ("copyMove", lambda out: apply_copy_move_forgery(args.input, out)),
        ("saltPepper", lambda out: apply_salt_and_pepper(args.input, out, args.sp_amount)),
        ("gaussianBlur", lambda out: apply_gaussian_blur(args.input, out, args.blur_kernel)),
        ("rotation", lambda out: apply_rotation(args.input, out, args.rotation_angle)),
        ("scaling", lambda out: apply_scaling(args.input, out, args.scale_factor)),
        ("histogramEqualization", lambda out: apply_histogram_equalization(args.input, out)),
    ]

    print(f"\nApplying attacks to: {args.input}")
    print(f"Output directory: {output_dir}\n")

    for attack_name, attack_func in attacks:
        output_path = os.path.join(output_dir, f"{base_name}_{attack_name}.png")
        try:
            attack_func(output_path)
            print(f"OK  {attack_name:25} -> {base_name}_{attack_name}.png")
        except Exception as e:
            print(f"ERR {attack_name:25} FAILED: {e}")

    print(f"\nAll attacked images saved in: {output_dir}/")


if __name__ == "__main__":
    main()
