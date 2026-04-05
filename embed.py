import cv2
import numpy as np

arnold_key = 10    
watermark_size = 128

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

host = cv2.imread("clock.png", cv2.IMREAD_GRAYSCALE)
host = host.astype(np.int32)
h, w = host.shape

wm = cv2.imread("watermark.png", cv2.IMREAD_GRAYSCALE)
wm = cv2.resize(wm, (watermark_size, watermark_size))
wm_scrambled = arnold_map(wm, arnold_key)

wm_bits = (wm_scrambled > 127).astype(np.uint8).flatten()
 
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

def compute_lbp_map(img):
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
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

lbp_map = compute_lbp_map(watermarked.astype(np.uint8))
np.save("lbp_map.npy", lbp_map)
print("LBP reference map generated and saved")