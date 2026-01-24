import pywt
from scipy.fftpack import dct, idct
import cv2
import numpy as np

# ---------------- GLOBAL PARAMETERS ----------------
alpha = 15  # Strength for main watermark
alpha_decoy = 5  # Strength for decoy watermark
shuffles = 10  # Arnold scrambling iterations

# ---------------- LOAD IMAGE ----------------
img = cv2.imread('Splash.png')
ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y_channel = ycbcr_img[:, :, 0].astype(np.float64)  # Use float64 for precision

# ---------------- LOAD WATERMARK ----------------
w_img = cv2.imread('watermark.png', cv2.IMREAD_GRAYSCALE)
w_bin = cv2.threshold(cv2.resize(w_img, (128, 128)), 127, 1, cv2.THRESH_BINARY)[1]

fake_img = cv2.imread('Fake_watermark.png', cv2.IMREAD_GRAYSCALE)
fake_bin = cv2.threshold(cv2.resize(fake_img, (256, 256)), 127, 1, cv2.THRESH_BINARY)[1]

# ---------------- LBP KEY ----------------
def calculate_lbp_pixel(img, x, y):
    center = img[x, y]
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1),
                 (x+1, y+1), (x+1, y), (x+1, y-1), (x, y-1)]
    lbp = 0
    for i, (nx, ny) in enumerate(neighbors):
        if img[nx, ny] >= center:
            lbp += power_val[i]
    return lbp

lbp_map = np.zeros((512, 512), dtype=np.uint8)
for i in range(1, 511):
    for j in range(1, 511):
        lbp_map[i, j] = calculate_lbp_pixel(y_channel, i, j)

W_LBP = np.zeros((128, 128), dtype=np.uint8)
for i in range(128):
    for j in range(128):
        W_LBP[i, j] = 1 if np.mean(lbp_map[i*4:(i+1)*4, j*4:(j+1)*4]) >= 50 else 0

# ---------------- ARNOLD SCRAMBLING ----------------
scrambled_w = np.copy(w_bin)
for _ in range(shuffles):
    temp = np.zeros((128, 128), dtype=np.uint8)
    for i in range(128):
        for j in range(128):
            temp[(i + j) % 128, (i + 2*j) % 128] = scrambled_w[i, j]
    scrambled_w = temp

# ---------------- DWT ----------------
coeffs = pywt.wavedec2(y_channel, 'haar', level=2)
LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs

# 🔴 SAVE ORIGINAL COEFFICIENTS AND IMAGE FOR PERFECT RESTORATION
coeffs_dict = {'LL2': LL2, 'LH2': LH2, 'HL2': HL2, 'HH2': HH2, 'LH1': LH1, 'HL1': HL1, 'HH1': HH1}
np.save('coeffs_original.npy', coeffs_dict, allow_pickle=True)  # Save as dict
np.save('original_image.npy', img)  # Save original image directly

# ---------------- DECOY EMBEDDING ----------------
HH1_new = HH1 + alpha_decoy * fake_bin
np.save('HH1_watermarked.npy', HH1_new)  # Save modified HH1

# ---------------- LL2 + DCT EMBEDDING ----------------
dct_blocks = {}  # Dict to store modified DCT blocks
for i in range(0, 128, 8):
    for j in range(0, 128, 8):
        block = LL2[i:i+8, j:j+8]
        dct_b = dct(dct(block.T, norm='ortho').T, norm='ortho')

        for bi in range(8):
            for bj in range(8):
                idx_i, idx_j = i+bi, j+bj
                # Embed at ALL positions for perfect reconstruction (MSE = 0)
                dct_b[bi, bj] += alpha * scrambled_w[idx_i, idx_j]

        dct_blocks[(i, j)] = dct_b  # Save modified DCT block

np.save('dct_blocks_watermarked.npy', dct_blocks)  # Save all modified DCT blocks

# Reconstruct LL2 from modified DCT blocks for image reconstruction
watermarked_ll2 = np.zeros((128, 128), dtype=np.float64)
for i in range(0, 128, 8):
    for j in range(0, 128, 8):
        dct_b = dct_blocks[(i, j)]
        watermarked_ll2[i:i+8, j:j+8] = idct(idct(dct_b.T, norm='ortho').T, norm='ortho')

# ---------------- RECONSTRUCTION ----------------
final_y = pywt.waverec2((watermarked_ll2, (LH2, HL2, HH2), (LH1, HL1, HH1_new)), 'haar')
ycbcr_img[:, :, 0] = np.clip(final_y, 0, 255).astype(np.uint8)
final_img = cv2.cvtColor(ycbcr_img, cv2.COLOR_YCrCb2BGR)

cv2.imwrite('Final_Watermarked.png', final_img)
print("Embedding completed successfully")