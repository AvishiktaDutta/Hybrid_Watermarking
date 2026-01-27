import cv2
import numpy as np
import sys

def lbp_pixel(img, x, y):
    center = img[x, y]
    neighbors = [
        img[x-1, y-1], img[x-1, y], img[x-1, y+1],
        img[x, y+1],
        img[x+1, y+1], img[x+1, y], img[x+1, y-1],
        img[x, y-1]
    ]
    code = 0
    for i, n in enumerate(neighbors):
        if n >= center:
            code |= (1 << i)
    return code
def compute_lbp_map(img):
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            lbp[i, j] = lbp_pixel(img, i, j)
    return lbp


def compute_lbp_key(lbp_map):
    key = np.zeros((128, 128), dtype=np.uint8)
    for i in range(128):
        for j in range(128):
            block = lbp_map[i*4:(i+1)*4, j*4:(j+1)*4]
            key[i, j] = 1 if np.mean(block) > 50 else 0
    return key

try:
    lbp_ref = np.load("lbp_key.npy")
except:
    print("Error: lbp_key.npy not found. Run embed stage first.")
    exit()

img_path = sys.argv[1] if len(sys.argv) > 1 else "watermarked.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: image not found")
    exit()

lbp_map = compute_lbp_map(img)
lbp_test = compute_lbp_key(lbp_map)

diff = lbp_ref != lbp_test
tampered_blocks = np.sum(diff)

print("Tampered blocks:", tampered_blocks)

if tampered_blocks == 0:
    print("RESULT: Image is authentic")
else:
    print("RESULT: Tampering detected")

    mask = np.zeros((512, 512), dtype=np.uint8)
    for i in range(128):
        for j in range(128):
            if diff[i, j]:
                mask[i*4:(i+1)*4, j*4:(j+1)*4] = 255

    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color[mask == 255] = [0, 0, 255]

    cv2.imwrite("tamper_mask.png", mask)
    cv2.imwrite("tamper_visualization.png", color)
    print("Tamper maps saved")
