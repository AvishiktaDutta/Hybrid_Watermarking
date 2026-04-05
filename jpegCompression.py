import cv2

def apply_jpeg_compression(image_path, output_path, quality_level=50):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Could not load image.")
        return

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_level]
    result = cv2.imwrite(output_path, img, encode_param)

    if result:
        print(f"JPEG Attack applied with Quality={quality_level}. Saved to {output_path}")
    else:
        print("Failed to save image.")

apply_jpeg_compression('watermarked.png', 'compressed_attack.jpg', quality_level=50)