# customCV.py>
print("customCV module loaded")

import cv2, struct
import numpy as np

def read_mnist(filePath):
    with open(filePath, 'rb') as f:
        # Read the header
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        # Read the image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        
    return images

def create_kernel():
    while True:
        choice = input("Use: \n<1> Rectangular \n<2> Circular \n")

        if choice in ["1", "2"]:
            break
        else:
            print("Choose a valid option.")

    while True:
        try:
            kernelWidth = int(input("Choose kernel width: "))
        except ValueError:
            print("Choose an integer value")
        else: 
            break

    if kernelWidth == 0:
        kernelWidth = 3

    while True:
        try:
            kernelHeight = int(input("Choose kernel height: "))
        except ValueError:
            print("Choose an integer value")
        else: 
            break
        
    if kernelHeight == 0:
        kernelHeight = 3

    if choice == "1":
        return np.ones((kernelHeight, kernelWidth), np.uint8), kernelHeight, kernelWidth
    else:
        kernel = np.zeros((kernelHeight, kernelWidth), dtype=np.int8)
    
        # Calculate the center of the ellipse
        centreX = kernelWidth // 2
        centreY = kernelHeight // 2
        
        # Define the radii of the ellipse (half of width and height)
        radiusX = kernelWidth // 2
        radiusY = kernelHeight // 2
        
        # Set elements inside the ellipse to 1 (or any desired value)
        for i in range(kernelHeight):
            for j in range(kernelWidth):
                if ((i - centreY) / radiusY)**2 + ((j - centreX) / radiusX)**2 <= 1:
                    kernel[i, j] = 1
        
        kernel[kernel == 0] = -1

        return kernel, kernelHeight, kernelWidth

def pad(img, kernelHeight, kernelWidth):
    padX = kernelWidth // 2
    padY = kernelHeight // 2
    paddedImage = np.pad(img, pad_width=((padY, padY), (padX, padX)), mode='constant', constant_values=0)

    return paddedImage

def erode(img, kernel, kernelHeight, kernelWidth):
    padX = kernelWidth // 2
    padY = kernelHeight // 2
    
    imgHeight, imgWidth = img.shape

    paddedImage = pad(img, kernelHeight, kernelWidth)
    erodedImage = np.zeros_like(img, dtype=np.uint8)

    for i in range(padY, imgHeight + padY):
        for j in range(padX, imgWidth + padX):
            roi = paddedImage[i-padY:i+padY+1, j-padX:j+padX+1]
            if np.all(roi[kernel == 1]):
                erodedImage[i-padY, j-padX] = 255

    return erodedImage


def dilate(img, kernel, kernelHeight, kernelWidth):
    padX = kernelWidth // 2
    padY = kernelHeight // 2

    imgHeight, imgWidth = img.shape
    
    paddedImage = pad(img, kernelHeight, kernelWidth)
    dilatedImage = np.zeros_like(img, dtype=np.uint8)

    for i in range(padY, imgHeight + padY):
        for j in range(padX, imgWidth + padX):
            roi = paddedImage[i-padY:i+padY+1, j-padX:j+padX+1]
            if np.any(roi[kernel > 0]):
                dilatedImage[i-padY, j-padX] = 255

    return dilatedImage

def opening(img, kernel, kernelHeight, kernelWidth):
    erodedImage = erode(img, kernel, kernelHeight, kernelWidth)
    openedImage = dilate(erodedImage, kernel, kernelHeight, kernelWidth)
    return openedImage

def closing(img, kernel, kernelHeight, kernelWidth):
    dilatedImage = dilate(img, kernel, kernelHeight, kernelWidth)
    closedImage = erode(dilatedImage, kernel, kernelHeight, kernelWidth)
    return closedImage