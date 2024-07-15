import cv2, struct
import numpy as np
from customCV import read_mnist

## Index in list is same as digit it represents
numberIndices = [10, 2, 1, 18, 4, 15, 11, 0, 61, 9]

def get_kernel():
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
        return cv2.getStructuringElement(cv2.MORPH_RECT, (kernelWidth, kernelHeight))
    else:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelWidth, kernelHeight))

def cv_dilate():
    print("Dilating each digit.")
    kernel = get_kernel()
    kernelHeight, kernelWidth = kernel.shape

    for i in range(len(numberIndices)):
        allNumbersImage[imgHeight:2*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.dilate(images[numberIndices[i]], kernel)

    cv2.namedWindow(f'All digits dilated, kernel ({kernelWidth},{kernelHeight})', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'All digits dilated, kernel ({kernelWidth},{kernelHeight})', 1400, 280)  
    cv2.imshow(f'All digits dilated, kernel ({kernelWidth},{kernelHeight})', allNumbersImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    test_loop()

def cv_erode():
    print("Eroding each digit.")
    kernel = get_kernel()
    kernelHeight, kernelWidth = kernel.shape

    for i in range(len(numberIndices)):
        allNumbersImage[imgHeight:2*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.erode(images[numberIndices[i]], kernel)

    cv2.imshow(f'All digits eroded, kernel ({kernelWidth},{kernelHeight})', allNumbersImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    test_loop()

def cv_opening():
    print("Opening each digit.")
    kernel = get_kernel()
    kernelHeight, kernelWidth = kernel.shape

    for i in range(len(numberIndices)):
        allNumbersImage[imgHeight:2*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.morphologyEx(images[numberIndices[i]], cv2.MORPH_OPEN, kernel)

    cv2.imshow(f'All digits opened, kernel ({kernelWidth},{kernelHeight})', allNumbersImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    test_loop()

def cv_closing():
    print("Closing each digit.")
    kernel = get_kernel()
    kernelHeight, kernelWidth = kernel.shape

    for i in range(len(numberIndices)):
        allNumbersImage[imgHeight:2*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.morphologyEx(images[numberIndices[i]], cv2.MORPH_CLOSE, kernel)

    cv2.imshow(f'All digits opened, kernel ({kernelWidth},{kernelHeight})', allNumbersImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    test_loop()

def test_loop():
    print("_____________________________________________________")
    print("Test openCV functions. Choose options: \n<1> Dilate \n<2> Erode \n<3> Open \n<4> Close \n<5> Exit Program")
    choice = input()
    if choice == "1":
        cv_dilate()
    elif choice == "2":
        cv_erode()
    elif choice == "3":
        cv_opening()
    elif choice == "4":
        cv_closing()
    elif choice == "5":
        exit()
    else:
        print(f"{choice} is not a valid option. \n")
        test_loop()    

## Open the MNIST dataset and read images
filePath = "MNIST_ORG/t10k-images.idx3-ubyte"
images = read_mnist(filePath)

## Create a grid of images
numImages, imgHeight, imgWidth = images.shape
# gridRows = gridCols = 25

# gridImage = np.zeros((gridRows * imgHeight, gridCols * imgWidth), dtype=np.uint8)

# for i in range(gridRows):
#     for j in range(gridCols):
#         imgIndex = i * 5 + j
#         if imgIndex < numImages:
#             # start y:end y, start x:end x
#             gridImage[i*imgHeight:(i+1)*imgHeight, j*imgWidth:(j+1)*imgWidth] = images[imgIndex]

# cv2.imshow('MNIST dataset grid', gridImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Getting one of each character
allNumbersImage = np.zeros((2*imgHeight, 10 * imgWidth), dtype=np.uint8)
for i in range(len(numberIndices)):
    allNumbersImage[0:imgHeight, i*imgWidth:(i+1)*imgWidth] = images[numberIndices[i]]


# cv2.imshow('One of each digit', allNumbersImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

test_loop()