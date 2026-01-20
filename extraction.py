import pywt
from scipy.fftpack import dct, idct
import cv2
import numpy as np

def extract_watermark():
    #1. Load the watermarked image
    watermarked_img = cv2.imread('Final_Watermarked.png')
    if watermarked_img is None:
        print("Error: Could not find the watermarked image!")
        return
    
    #2. Convert to YCbCr and isolate Y channel
    ycbcr_ext = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YCrCb)
    y_watermarked = ycbcr_ext[:, :, 0]

    #3. Apply 2-level DWT to find the LL2 subband
    coeffs = pywt.wavedec2(y_watermarked, 'haar', level=2)
    LL2_ext = coeffs[0]
    details = coeffs[1:]

    #4.Extract bits using the same 8x8 DCT blocks
    recovered_w = np.zeros((128, 128), dtype=np.uint8)
    recovered_lbp = np.zeros((128, 128), dtype=np.uint8)
    recovered_ll2 = np.zeros((128, 128), dtype=np.float32)
    
    for i in range(0, 128, 8):
        for j in range(0, 128, 8):
            block = LL2_ext[i: i+8, j: j+8]
            dct_block = dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

            #Use the coordinates where you embedded the data
            for bi in range(8):
                for bj in range(8):
                    #Map back to 128x128 grid
                    idx_i, idx_j = i+bi, j+bj
                    val = int(round(dct_block[bi, bj]))

                    #Extract using(reverse PEE)
                    combined_bits = val%4
                    recovered_lbp[idx_i, idx_j] = (combined_bits >> 1) & 1
                    recovered_w[idx_i, idx_j] = combined_bits & 1
                    #substract the hidden bits to get the original Dct coefficients
                    original_val = val - combined_bits
                    #store this in a new array to recistruct the original image
                    recovered_ll2[idx_i, idx_j] = dct_block[bi, bj] - combined_bits
    
    #Reconstruct the original hoost image
    original_ll2_pixels = np.zeros((128, 128), dtype=np.float32)
    for i in range(0, 128, 8):
        for j in range(0, 128, 8):
            block = recovered_ll2[i: i+8, j: j+8]
            original_ll2_pixels[i: i+8, j: j+8] = idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

    new_coeffs = [original_ll2_pixels] + details
    recovered_y = pywt.waverec2(new_coeffs, 'haar')
    recovered_y = np.clip(recovered_y, 0, 255).astype(np.uint8)

    #5. Descrambling (Inverse Arnold Cat Map)
    N = 128
    shuffles = 10
    descrambled_w = np.copy(recovered_w)
    
    for _ in range(shuffles):
        temp_img = np.zeros((N, N), dtype=np.uint8) 
        for r in range(N):
            for c in range(N):
                ni = (2 * r - c) % N
                nj = (-r + c) % N
                temp_img[ni, nj] = descrambled_w[r, c]
        descrambled_w = np.copy(temp_img)
    cv2.imshow('Recovered host Y', recovered_y)
    cv2.imshow('Extracted Watermark', descrambled_w * 255)
    cv2.imwrite('Extracted_Watermark.png', descrambled_w * 255)
    cv2.imshow('Extracted LBP Key', recovered_lbp * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
 extract_watermark()
