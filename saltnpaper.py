import cv2
import numpy as np
import random

def add_salt_and_pepper_noise(image, amount=0.02):
    
    output = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.1)
    num_pepper = np.ceil(amount * image.size * 0.1)

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    output[tuple(coords)] = 255

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    output[tuple(coords)] = 0
    
    return output

# Usage
watermarked_img = cv2.imread('watermarked.png', 0)
attacked_img = add_salt_and_pepper_noise(watermarked_img, amount=0.05)
cv2.imwrite('attacked_image.png', attacked_img)