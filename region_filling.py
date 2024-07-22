import cv2, struct
import numpy as np
from customCV import read_mnist, create_kernel, erode, dilate, opening, closing, fill_holes, binarize, find_holes

## Index in list is same as digit it represents
numberIndices = [10, 2, 1, 18, 4, 15, 11, 0, 61, 9]

filePath = "MNIST_ORG/t10k-images.idx3-ubyte"
images = read_mnist(filePath)

## Create a grid of images
numImages, imgHeight, imgWidth = images.shape

allNumbersImage = np.zeros((4*imgHeight, 10 * imgWidth), dtype=np.uint8)

images = np.array([binarize(img) for img in images])

holesImage = find_holes(images[61])

cv2.namedWindow(f'filled', cv2.WINDOW_NORMAL)
cv2.resizeWindow(f'filled', 840, 840)  
cv2.imshow(f'filled', holesImage)
cv2.waitKey(0)