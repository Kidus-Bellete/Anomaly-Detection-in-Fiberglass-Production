# utils/preprocessing.py
import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, img_size=256):
        self.img_size = img_size
    
    def enhance_defects(self, image):
        """Enhance potential defect features in the image"""
        # Convert to float32
        image = image.astype(np.float32) / 255.0
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            image[:,:,0] = clahe.apply(np.uint8(image[:,:,0] * 255)) / 255.0
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        else:
            image = clahe.apply(np.uint8(image * 255)) / 255.0
        
        # Edge enhancement
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        enhanced = cv2.filter2D(image, -1, kernel)
        
        return enhanced

