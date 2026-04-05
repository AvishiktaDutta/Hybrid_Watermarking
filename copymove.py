import cv2
import numpy as np

def apply_copy_move_forgery(image_path, output_path, start_coords, end_coords, target_coords):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find image at {image_path}")
        return
    
    y1, x1 = start_coords
    y2, x2 = end_coords
    source_block = img[y1:y2, x1:x2].copy()
    h, w = source_block.shape[:2]
    ty, tx = target_coords

    img[ty:ty+h, tx:tx+w] = source_block

    cv2.imwrite(output_path, img)
    print(f"Invisible forgery saved as {output_path}")

apply_copy_move_forgery(
    image_path='watermarked.png', 
    output_path='duplicated_frame_forgery.png',
    start_coords=(125, 15),
    end_coords=(185, 90), 
    target_coords=(125, 100) 
)