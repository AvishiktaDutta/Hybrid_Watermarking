import cv2
import numpy as np

ARNOLD_ITERS = 10    
WM_SIZE = 128

# INVERSE ARNOLD 
def inverse_arnold_map(img, iters):
    n = img.shape[0]
    out = img.copy()
    for _ in range(iters):
        temp = np.zeros_like(out)
        for x in range(n):
            for y in range(n):
                nx = (2*x - y) % n
                ny = (-x + y) % n
                temp[nx, ny] = out[x, y]
        out = temp
    return out

#LOAD WATERMARKED IMAGE 
wm_img = cv2.imread("watermarked.png", cv2.IMREAD_GRAYSCALE)
wm_img = wm_img.astype(np.int32)
h, w = wm_img.shape

location_map = np.load("location_map.npy")
payload_len = int(np.load("payload_len.npy"))

restored = wm_img.copy()
bits = []

lm_idx = 0
bit_idx = 0

#  DIFFERENCE EXPANSION EXTRACTION
for i in range(0, h, 2):
    for j in range(0, w - 1, 2):

        if lm_idx >= len(location_map):
            break

        flag = location_map[lm_idx]
        lm_idx += 1

        if flag == 0:
            continue

        x = int(wm_img[i, j])
        y = int(wm_img[i, j+1])

        d_new = x - y

        if bit_idx < payload_len:
            bit = d_new & 1
            bits.append(bit)
            bit_idx += 1

        d = d_new // 2
        a = (x + y) // 2

        x_orig = a + (d + 1)//2
        y_orig = a - d//2

        restored[i, j]   = x_orig
        restored[i, j+1] = y_orig

# WATERMARK RECOVERY 
bits = np.array(bits, dtype=np.uint8)
wm_scrambled = bits[:WM_SIZE*WM_SIZE].reshape((WM_SIZE, WM_SIZE)) * 255

# Apply inverse Arnold
wm_final = inverse_arnold_map(wm_scrambled.astype(np.uint8), ARNOLD_ITERS)

cv2.imwrite("restored_host.png", restored.astype(np.uint8))
cv2.imwrite("extracted_watermark.png", wm_final)

print("Extraction completed with Arnold descrambling")
