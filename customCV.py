# customCV.py>
print("customCV module loaded")

import cv2, struct
import numpy as np

def read_mnist(filePath):
    print("opening")
    with open(filePath, 'rb') as f:
        # Read the header
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        # Read the image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        
    return images

def pad(img, padding):
    pass

def erode(img, kernel):
    pass

def dilate(img, kernel):
    pass

def opening(img, kernel):
    pass

def closing(img, kernel):
    pass