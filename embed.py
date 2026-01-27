import cv2
import numpy as np

#PARAMETERS
ARNOLD_ITERS = 10    
WM_SIZE = 128

# ARNOLD CAT MAP 
def arnold_map(img, iters):
    n = img.shape[0]
    out = img.copy()
    for _ in range(iters):
        temp = np.zeros_like(out)
        for x in range(n):
            for y in range(n):
                nx = (x + y) % n
                ny = (x + 2*y) % n
                temp[nx, ny] = out[x, y]
        out = temp
    return out

# LOAD HOST
host = cv2.imread("boat.png", cv2.IMREAD_GRAYSCALE)
host = host.astype(np.int32)
h, w = host.shape

# LOAD WATERMARK
wm = cv2.imread("watermark.png", cv2.IMREAD_GRAYSCALE)
wm = cv2.resize(wm, (WM_SIZE, WM_SIZE))

# Apply Arnold scrambling
wm_scrambled = arnold_map(wm, ARNOLD_ITERS)

# Convert to binary bitstream
wm_bits = (wm_scrambled > 127).astype(np.uint8).flatten()

#DIFFERENCE EXPANSION 
watermarked = host.copy()
location_map = []
idx = 0

for i in range(0, h, 2):
    for j in range(0, w - 1, 2):
        if idx >= len(wm_bits):
            break

        x = int(host[i, j])
        y = int(host[i, j+1])

        d = x - y
        a = (x + y) // 2
        bit = int(wm_bits[idx])

        d_new = 2*d + bit
        x_new = a + (d_new + 1)//2
        y_new = a - d_new//2

        if 0 <= x_new <= 255 and 0 <= y_new <= 255:
            watermarked[i, j]   = x_new
            watermarked[i, j+1] = y_new
            location_map.append(1)
            idx += 1
        else:
            location_map.append(0)

    if idx >= len(wm_bits):
        break

np.save("location_map.npy", np.array(location_map, dtype=np.uint8))
np.save("payload_len.npy", idx)

cv2.imwrite("watermarked.png", watermarked.astype(np.uint8))

print("Embedding completed with Arnold scrambling")
print("Embedded bits:", idx)

def calculate_lbp_pixel(img, x, y):
    center = img[x, y]
    neighbors = [
        img[x-1, y-1], img[x-1, y], img[x-1, y+1],
        img[x, y+1],
        img[x+1, y+1], img[x+1, y], img[x+1, y-1],
        img[x, y-1]
    ]
    lbp = 0
    for i, n in enumerate(neighbors):
        if n >= center:
            lbp |= (1 << i)
    return lbp


def compute_lbp_map(img):
    h, w = img.shape
    lbp_map = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            lbp_map[i, j] = calculate_lbp_pixel(img, i, j)
    return lbp_map


def compute_lbp_key(lbp_map):
    key = np.zeros((128, 128), dtype=np.uint8)
    for i in range(128):
        for j in range(128):
            block = lbp_map[i*4:(i+1)*4, j*4:(j+1)*4]
            key[i, j] = 1 if np.mean(block) >= 50 else 0
    return key


# Convert watermarked image to uint8 for LBP
watermarked_uint8 = watermarked.astype(np.uint8)

# Compute LBP reference
lbp_map = compute_lbp_map(watermarked_uint8)
lbp_key = compute_lbp_key(lbp_map)

# Save LBP reference data
np.save("lbp_map.npy", lbp_map)
np.save("lbp_key.npy", lbp_key)

print("LBP reference generated for tamper detection")
