import cv2
import numpy as np
import sys

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

lbp_ref = np.load("lbp_map.npy")

img_path = sys.argv[1] if len(sys.argv) > 1 else "attacked_image.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

lbp_test = compute_lbp_map(img)

diff = lbp_ref != lbp_test
tampered_pixels = np.sum(diff)

print("Tampered pixels:", tampered_pixels)

if tampered_pixels == 0:
    print("Image is authentic")
else:
    print("Tampering detected")
    mask = (diff * 255).astype(np.uint8)
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[mask == 255] = [0, 0, 255]

    cv2.imwrite("tamper_mask.png", mask)
    cv2.imwrite("tamper_visualization.png", overlay)