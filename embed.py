import pywt
from scipy.fftpack import dct, idct
import cv2
import numpy as np
#STEP-1
#Load image(512 x 512)
img = cv2.imread('Splash.png')
#Convert color space RGB to yCbCr
#OpenCv loads image in RGB by default so we use BGR2YCrCb
ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#Isolate the y channel (Luminance)
y_channel = ycbcr_img[:, :, 0]
#Display the y channel
#cv2.imshow('Y Channel', y_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

#STEP-2 
#generate LBP key
def get_lbp_bit(image_block, threshold_val):
    #Calculate the avg lbp of each 4x4 block
    avg_lbp = np.mean(image_block)
    #Return 1 if texture is high , 0 if flat
    return 1 if avg_lbp>=0 else 0

#create a function to calculate a simple LBP for a pixel
def calculate_lbp_pixel(img, x, y):
    center = img[x, y]
    #weights for 8 neighbours
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    lbp_val = 0

    #define neighbour coordinates relative to centre (clockwise)
    neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1),
              (x+1, y+1), (x+1, y), (x+1, y-1), (x, y-1)]
    for i ,(nx, ny) in enumerate(neighbors): 
        if img[nx, ny]>= center:
            lbp_val += power_val[i]
        return lbp_val
lbp_map = np.zeros((512, 512), dtype = np.uint8)
for i in range(1, 511):
    for j in range(1, 511):
        lbp_map[i, j] = calculate_lbp_pixel(y_channel, i, j)

W_LBP = np.zeros((128, 128), dtype = np.uint8)
T_LBP = 50

for i in range(128):
    for j in range(128):
        image_block = lbp_map[i*4: (i+1)*4, j*4: (j+1)*4]
        W_LBP[i, j] = get_lbp_bit(image_block, T_LBP)

#STEP-3
#Arnold SCrambling
watermark_img = cv2.imread('watermark.png', cv2.IMREAD_GRAYSCALE)
watermark_resized = cv2.resize(watermark_img, (128, 128))
_, binary_watermark = cv2.threshold(watermark_resized, 127, 1, cv2.THRESH_BINARY)
N = 128
shuffles = 10
# 2. get a copy so we do not need to change the original
scrambled_w = np.copy(binary_watermark)

# 3. Perform the suffle 'iterations' times
for count in range(shuffles):
        temp_img = np.zeros((N, N), dtype=np.uint8)
        for i in range(N):
            for j in range(N):
                # The cat map formula
                ni = (i + j) % N
                nj = (i + 2*j)% N
                temp_img[ni, nj] = scrambled_w[i, j]
            scrambled_w = np.copy(temp_img)

print("Watermark loaded and scrambled")

# Step-4 DWT and DCT Decomposition
# Apply 2-level DWT on the Y-Channel
# This shrinks the 512x512 image into a 128x128 'LL2' sub-band
coeffs = pywt.wavedec2(y_channel, 'haar', level=2)
LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs

# 2. Divide LL2 into 8x8 blocks and apply DCT
# LL2 is 128x128, so it contains 16x16 blocks of size 8x8

N_LL2 = 128
block_size = 8

#Create an empty map store out DCT results
ll2_dct_map = np.zeros((N_LL2, N_LL2), dtype = np.float32)
for i in range(0, N_LL2, block_size):
    for j in range(0, N_LL2, block_size):
        #define the 8x8 block
        block = LL2[i : i+block_size, j : j+block_size]

        #apply 2D-dct to the block
        #do it twice: once for rows, once for columns
        dct_block = dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

        #save the frequencies into our map
        ll2_dct_map[i : i+block_size, j: j+block_size] = dct_block
print("Step-4 complete : Image transformed into frequency blocks.")

#Dual-Layer PEE Embedding
watermarked_dct = np.copy(ll2_dct_map)

#We use two nested loops to go through every bit(128x128)
for i in range(128):
    for j in range(128):
        #1. Select two coefficients(x1 and x2) from the dct map
        #We will use neighboring coefficients for the PEE pair
        x1 = ll2_dct_map[i, j]
        #In a real system, x2 would be a specific neighbor.
        #For this step, let's use a small offset to represent the pair:
        x2 = x1 + 0.001

        # 2. Get the two bits
        w_main = scrambled_w[i, j]
        w_lbp = W_LBP[i, j]

        #3. Apply formula
        e = x2 - x1
        e_prime = (4 * e) + (2 * w_lbp) + w_main

        #4. update the image with the hidden data
        watermarked_dct[i, j] = x1 + e_prime
print("Embedding successfully completed!")

#1. Reverse the DCT(Inverse DCT)
# This turns the frequencies back into LL2 pixels
final_ll2 = np.zeros((128, 128), dtype=np.float32)
for i in range(0, 128, 8):
    for j in range(0, 128, 8):
        block = watermarked_dct[i:i+8, j:j+8]
        final_ll2[i:i+8, j:j+8] = idct(idct(block.T, norm='ortho').T, norm='ortho')

#2. Reverse the DWT(Inverse Dwt)
#This combines LL2 with the other parts to make a 512x512 image
coeffs_new = (final_ll2, (LH2, HL2, HH2), (LH1, HL1, HH1))
watermarked_y = pywt.waverec2(coeffs_new, 'haar')

watermarked_y = np.clip(watermarked_y, 0, 255).astype(np.uint8)

#3. put the Y channel back into the original color image
ycbcr_img[:, :, 0] = watermarked_y
# Convert back to the normal BGR color space for viewing
final_output = cv2.cvtColor(ycbcr_img, cv2.COLOR_YCrCb2BGR)

# Save the successful result
cv2.imwrite('Final_Watermarked.png', final_output)
cv2.imshow('Embedded successfully', final_output)
cv2.waitKey(0)