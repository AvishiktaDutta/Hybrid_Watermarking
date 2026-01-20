import pywt
from scipy.fftpack import dct, idct
import cv2
import numpy as np

def extract_watermark():
    # --- 1. GLOBAL PARAMETERS (Must match embed.py) ---
    alpha = 2
    alpha_decoy = 5
    shuffles = 10
    
    # --- 2. LOAD & PRE-PROCESS ---
    # Load as float32 to preserve the 0.1 alpha shifts
    img = cv2.imread('Final_Watermarked.png')
    if img is None:
        print("Error: Image not found.")
        return
        
    y_ext = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    
    # --- 3. DWT DECOMPOSITION ---
    coeffs = pywt.wavedec2(y_ext, 'haar', level=2)
    LL2_ext = coeffs[0]
    HH1_ext = coeffs[2][2] # Get HH1 subband for decoy

    # --- 4. ROBUST DECOY EXTRACTION (HH1) ---
    # Isolate the signal and use a relative threshold
    decoy_bin = (HH1_ext > (alpha_decoy / 2)).astype(np.uint8)

    
    # NEW: Apply a 3x3 Median Filter to remove the 'salt and pepper' noise
    decoy_final = cv2.medianBlur((decoy_bin * 255), 3)

    # --- 5. REVERSIBLE EXTRACTION (LL2) ---
    recovered_w = np.zeros((128, 128), dtype=np.uint8)
    recovered_lbp = np.zeros((128, 128), dtype=np.uint8)
    
    for i in range(0, 128, 8):
        for j in range(0, 128, 8):
            block = LL2_ext[i:i+8, j:j+8]
            dct_b = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            for bi in range(8):
                for bj in range(8):
                    y = dct_b[bi, bj]
                    y_int = int(np.round(y))
                    
                    # ISOLATE FRACTION: Correctly handle the 0.1 alpha
                    # This recovers the integer (0-3) we embedded
                    diff = y_int % 8
                    bits = diff // alpha
                    
                    recovered_w[i+bi, j+bj] = bits & 1         # Main Bit
                    recovered_lbp[i+bi, j+bj] = (bits >> 1) & 1 # LBP Bit

    # 6. ARNOLD DESCRAMBLING 
    for _ in range(shuffles):
        # 1. Create a fresh blank grid for this iteration
        temp = np.zeros((128, 128), dtype=np.uint8)
        
        for r in range(128):
            for c in range(128):
                # 2. THE CORRECT INVERSE FORMULA
                # To reverse (i+j, i+2j), the math must be:
                ni = (2 * r - c) % 128
                nj = (-r + c) % 128
                temp[ni, nj] = recovered_w[r, c]
        
        # 3. CRITICAL: This line must be aligned with "for r"
        # This saves the current iteration's result for the next iteration
        recovered_w = np.copy(temp)
    # --- 7. DISPLAY RESULTS ---
    decoy_final = cv2.medianBlur((decoy_bin * 255).astype(np.uint8), 3)
    cv2.imshow('Decoy Watermark (HH1)', decoy_final)
    cv2.imshow('Main Watermark (LL2)', recovered_w * 255)
    cv2.imshow('Extracted LBP Key', recovered_lbp * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_watermark()