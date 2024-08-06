import cv2, struct
import numpy as np
from customCV import *

## Index in list is same as digit it represents
numberIndices = [10, 2, 1, 18, 4, 15, 11, 0, 61, 9]

## Default text parametres
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (50, 50, 255)  # White color in BGR format
thickness = 1

filePath = "MNIST_ORG/t10k-images.idx3-ubyte"
images = read_mnist(filePath)
images = np.array([binarize(img) for img in images])

## Create a grid of images
numImages, imgHeight, imgWidth = images.shape

allNumbersImage = np.zeros((4*imgHeight, 10 * imgWidth), dtype=np.uint8)
for i in range(len(numberIndices)):
    allNumbersImage[imgHeight:2*imgHeight, i*imgWidth:(i+1)*imgWidth] = images[numberIndices[i]]

text = "Original Digits:"
org = (imgWidth, imgHeight - 2)
cv2.putText(allNumbersImage, text, org, font, font_scale, font_color, thickness)

for i in range(len(numberIndices)):
    x1, y1, x2, y2 = bounding_box(images[numberIndices[i]])
    allNumbersImage[3*imgHeight:4*imgHeight, i*imgWidth:(i+1)*imgWidth] = draw_rect(images[numberIndices[i]], x1, y1, x2, y2)

text = "Bounded Digits:"
org = (imgWidth, 3*imgHeight - 2)
cv2.putText(allNumbersImage, text, org, font, font_scale, font_color, thickness)

cv2.namedWindow(f'Bounded digits', cv2.WINDOW_NORMAL)
cv2.resizeWindow(f'Bounded digits', 1400, 560)  
cv2.imshow(f'Bounded digits', allNumbersImage)
cv2.waitKey(0)