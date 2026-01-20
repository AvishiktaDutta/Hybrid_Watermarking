import pywt
from scipy.fftpack import dct, idct
import cv2
import numpy as np
#STEP-1
#Global parameters
alpha = 2
alpha_decoy = 5
T_max = 20
shuffles = 10

#Load image(512 x 512)
img = cv2.imread('Splash.png')
#Convert color space RGB to ycbc_img
#OpenCv loads image in RGB by default so we use BGR2YCrCb
ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#Isolate the y channel (Luminance)
y_channel = ycbcr_img[:, :, 0].astype(np.float32)
#Primary Watermark(128x128)
w_img = cv2.imread('watermark.png', cv2.IMREAD_GRAYSCALE)
w_bin = cv2.threshold(cv2.resize(w_img, (128, 128)), 127, 1, cv2.THRESH_BINARY)[1]
#Fake Watermark(128x128)
fake_img = cv2.imread('Fake_watermark.png', cv2.IMREAD_GRAYSCALE)
fake_bin = cv2.threshold(cv2.resize(fake_img, (256, 256)), 127, 1, cv2.THRESH_BINARY)[1]
#STEP-2 
#generate LBP key
def calculate_lbp_pixel(img, x, y):
    center = img[x, y]
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    lbp_val = 0
    neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1),
                 (x+1, y+1), (x+1, y), (x+1, y-1), (x, y-1)]
    for i, (nx, ny) in enumerate(neighbors):
        if img[nx, ny] >= center: lbp_val += power_val[i]
    return lbp_val

lbp_map = np.zeros((512, 512), dtype=np.uint8)
for i in range(1, 511):
    for j in range(1, 511):
        lbp_map[i, j] = calculate_lbp_pixel(y_channel, i, j)

W_LBP = np.zeros((128, 128), dtype=np.uint8)
for i in range(128):
    for j in range(128):
        W_LBP[i, j] = 1 if np.mean(lbp_map[i*4:(i+1)*4, j*4:(j+1)*4]) >= 50 else 0

# 3. ARNOLD SCRAMBLING
scrambled_w = np.copy(w_bin)
for _ in range(shuffles):
    temp = np.zeros((128, 128), dtype=np.uint8)
    for i in range(128):
        for j in range(128):
            temp[(i + j) % 128, (i + 2*j) % 128] = scrambled_w[i, j]
    scrambled_w = np.copy(temp)

# 4. DWT & DCT DECOMPOSITION
coeffs = pywt.wavedec2(y_channel, 'haar', level=2)
LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs

# 5. DUAL-LAYER EMBEDDING
# A. HH1 Decoy (Security layer)
HH1_new = HH1 + (alpha_decoy * fake_bin)

# B. LL2 PEE Embedding
watermarked_ll2_dct = np.zeros((128, 128), dtype=np.float32)
for i in range(0, 128, 8):
    for j in range(0, 128, 8):
        block = LL2[i:i+8, j:j+8]
        dct_b = dct(dct(block.T, norm='ortho').T, norm='ortho')
        for bi in range(8):
            for bj in range(8):
                idx_i, idx_j = i+bi, j+bj
                x1 = dct_b[bi, bj]
                e = 0.001 # Small proxy for prediction error
                if abs(e) < T_max:
                    e_p = (4 * e) + (2 * W_LBP[idx_i, idx_j]) + scrambled_w[idx_i, idx_j]
                    dct_b[bi, bj] = x1 + (alpha * e_p)
                else:
                    dct_b[bi, bj] = x1 + e + T_max
        watermarked_ll2_dct[i:i+8, j:j+8] = idct(idct(dct_b.T, norm='ortho').T, norm='ortho')

# 6. RECONSTRUCTION
final_y = pywt.waverec2((watermarked_ll2_dct, (LH2, HL2, HH2), (LH1, HL1, HH1_new)), 'haar')
ycbcr_img[:, :, 0] = np.clip(final_y, 0, 255).astype(np.uint8)
final_img = cv2.cvtColor(ycbcr_img, cv2.COLOR_YCrCb2BGR)
cv2.imwrite('Final_Watermarked.png', final_img)
print("Embedding finished perfectly.")