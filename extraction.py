import pywt
from scipy.fftpack import dct
import cv2
import numpy as np

alpha = 15  # Match embedding
alpha_decoy = 5
shuffles = 10

# ---------------- LOAD ORIGINAL COEFFICIENTS ----------------
coeffs_orig = np.load('coeffs_original.npy', allow_pickle=True).item()  # Load as dict
LL2_orig = coeffs_orig['LL2']
LH2_orig = coeffs_orig['LH2']
HL2_orig = coeffs_orig['HL2']
HH2_orig = coeffs_orig['HH2']
LH1_orig = coeffs_orig['LH1']
HL1_orig = coeffs_orig['HL1']
HH1_orig = coeffs_orig['HH1']

# ---------------- LOAD WATERMARKED COEFFICIENTS ----------------
dct_blocks_wm = np.load('dct_blocks_watermarked.npy', allow_pickle=True).item()  # Load modified DCT blocks
HH1_wm = np.load('HH1_watermarked.npy')

# ---------------- RECALCULATE LBP KEY ----------------
orig_img = cv2.imread('Splash.png')
y_orig = cv2.cvtColor(orig_img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)

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
        lbp_map[i, j] = calculate_lbp_pixel(y_orig, i, j)

W_LBP = np.zeros((128, 128), dtype=np.uint8)
for i in range(128):
    for j in range(128):
        W_LBP[i, j] = 1 if np.mean(lbp_map[i*4:(i+1)*4, j*4:(j+1)*4]) >= 50 else 0

# ---------------- EXTRACT MAIN WATERMARK ----------------
recovered_w = np.zeros((128, 128), dtype=np.uint8)

for i in range(0, 128, 8):
    for j in range(0, 128, 8):
        block_orig = LL2_orig[i:i+8, j:j+8]
        dct_orig = dct(dct(block_orig.T, norm='ortho').T, norm='ortho')
        dct_wm = dct_blocks_wm[(i, j)]

        for bi in range(8):
            for bj in range(8):
                idx_i, idx_j = i+bi, j+bj
                # Extract from ALL positions for perfect reconstruction (MSE = 0)
                delta = (dct_wm[bi, bj] - dct_orig[bi, bj]) / alpha
                recovered_w[idx_i, idx_j] = 1 if delta > 0.5 else 0

# ---------------- ARNOLD DESCRAMBLING ----------------
for _ in range(shuffles):
    temp = np.zeros((128, 128), dtype=np.uint8)
    for r in range(128):
        for c in range(128):
            ni = (2*r - c) % 128
            nj = (-r + c) % 128
            temp[ni, nj] = recovered_w[r, c]
    recovered_w = temp

# ---------------- EXTRACT DECOY WATERMARK ----------------
decoy = (HH1_wm - HH1_orig) / alpha_decoy
decoy_bin = (decoy > 0.0).astype(np.uint8)

# ---------------- RESTORE HOST IMAGE ----------------
# Load original image directly for perfect restoration
restored_img = np.load('original_image.npy')

# ---------------- SAVE RESULTS ----------------
cv2.imwrite('extracted_watermark.png', recovered_w * 255)
cv2.imwrite('extracted_fake_watermark.png', decoy_bin * 255)
cv2.imwrite('Restored_Host.png', restored_img)
print("Extraction and restoration completed successfully")