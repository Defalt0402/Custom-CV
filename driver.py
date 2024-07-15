import cv2, struct
import numpy as np
from customCV import read_mnist, create_kernel, erode, dilate, opening, closing

## Index in list is same as digit it represents
numberIndices = [10, 2, 1, 18, 4, 15, 11, 0, 61, 9]

## Default text parametres
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (50, 50, 255)  # White color in BGR format
thickness = 1

comparing = False


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

    if not comparing:
        for i in range(len(numberIndices)):
            allNumbersImage[3*imgHeight:4*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.dilate(images[numberIndices[i]], kernel)

        text = "Dilated Digits:"
        org = (imgWidth, 3*imgHeight - 2)
        cv2.putText(allNumbersImage, text, org, font, font_scale, font_color, thickness)


        cv2.namedWindow(f'All digits dilated, kernel ({kernelWidth},{kernelHeight})', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'All digits dilated, kernel ({kernelWidth},{kernelHeight})', 1400, 560)  
        cv2.imshow(f'All digits dilated, kernel ({kernelWidth},{kernelHeight})', allNumbersImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        allNumbersImage[2*imgHeight:3*imgHeight, 0:10*imgWidth] = 0

        test_loop()
    else:
        for i in range(len(numberIndices)):
            comparisonImage[3*imgHeight:4*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.dilate(images[numberIndices[i]], kernel)

        text = "openCV Dilated Digits:"
        org = (imgWidth, 3*imgHeight - 2)
        cv2.putText(comparisonImage, text, org, font, font_scale, font_color, thickness)

def cv_erode():
    print("Eroding each digit.")
    kernel = get_kernel()
    kernelHeight, kernelWidth = kernel.shape

    if not comparing:
        for i in range(len(numberIndices)):
            allNumbersImage[3*imgHeight:4*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.erode(images[numberIndices[i]], kernel)

        text = "Eroded Digits:"
        org = (imgWidth, 3*imgHeight - 2)
        cv2.putText(allNumbersImage, text, org, font, font_scale, font_color, thickness)

        cv2.namedWindow(f'All digits eroded, kernel ({kernelWidth},{kernelHeight})', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'All digits eroded, kernel ({kernelWidth},{kernelHeight})', 1400, 560)  
        cv2.imshow(f'All digits eroded, kernel ({kernelWidth},{kernelHeight})', allNumbersImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        allNumbersImage[2*imgHeight:3*imgHeight, 0:10*imgWidth] = 0

        test_loop()
    else:
        for i in range(len(numberIndices)):
            comparisonImage[3*imgHeight:4*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.erode(images[numberIndices[i]], kernel)

        text = "openCV Eroded Digits:"
        org = (imgWidth, 3*imgHeight - 2)
        cv2.putText(comparisonImage, text, org, font, font_scale, font_color, thickness)

def cv_opening():
    print("Opening each digit.")
    kernel = get_kernel()
    kernelHeight, kernelWidth = kernel.shape

    if not comparing:
        for i in range(len(numberIndices)):
            allNumbersImage[3*imgHeight:4*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.morphologyEx(images[numberIndices[i]], cv2.MORPH_OPEN, kernel)

        text = "Opened Digits:"
        org = (imgWidth, 3*imgHeight - 2)
        cv2.putText(allNumbersImage, text, org, font, font_scale, font_color, thickness)

        cv2.namedWindow(f'All digits opened, kernel ({kernelWidth},{kernelHeight})', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'All digits opened, kernel ({kernelWidth},{kernelHeight})', 1400, 560)  
        cv2.imshow(f'All digits opened, kernel ({kernelWidth},{kernelHeight})', allNumbersImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        allNumbersImage[2*imgHeight:3*imgHeight, 0:10*imgWidth] = 0


        test_loop()
    else:
        for i in range(len(numberIndices)):
            comparisonImage[3*imgHeight:4*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.morphologyEx(images[numberIndices[i]], cv2.MORPH_OPEN, kernel)

        text = "openCV Opened Digits:"
        org = (imgWidth, 3*imgHeight - 2)
        cv2.putText(comparisonImage, text, org, font, font_scale, font_color, thickness)

def cv_closing():
    print("Closing each digit.")
    kernel = get_kernel()
    kernelHeight, kernelWidth = kernel.shape

    if not comparing:
        for i in range(len(numberIndices)):
            allNumbersImage[3*imgHeight:4*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.morphologyEx(images[numberIndices[i]], cv2.MORPH_CLOSE, kernel)

        text = "Closed Digits:"
        org = (imgWidth, 3*imgHeight - 2)
        cv2.putText(allNumbersImage, text, org, font, font_scale, font_color, thickness)

        cv2.namedWindow(f'All digits closed, kernel ({kernelWidth},{kernelHeight})', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'All digits closed, kernel ({kernelWidth},{kernelHeight})', 1400, 560)  
        cv2.imshow(f'All digits closed, kernel ({kernelWidth},{kernelHeight})', allNumbersImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        allNumbersImage[2*imgHeight:3*imgHeight, 0:10*imgWidth] = 0

        test_loop()
    else:
        for i in range(len(numberIndices)):
            comparisonImage[3*imgHeight:4*imgHeight, i*imgWidth:(i+1)*imgWidth] = cv2.morphologyEx(images[numberIndices[i]], cv2.MORPH_CLOSE, kernel)

        text = "openCV Closed Digits:"
        org = (imgWidth, 3*imgHeight - 2)
        cv2.putText(comparisonImage, text, org, font, font_scale, font_color, thickness)

def compare_dilate():
    cv_dilate()
    kernel, kernelHeight, kernelWidth = create_kernel()
    for i in range(len(numberIndices)):
        comparisonImage[5*imgHeight:6*imgHeight, i*imgWidth:(i+1)*imgWidth] = dilate(images[numberIndices[i]], kernel, kernelHeight, kernelWidth)
    
    text = "Custom Dilated Digits:"
    org = (imgWidth, 5*imgHeight - 2)
    cv2.putText(comparisonImage, text, org, font, font_scale, font_color, thickness)

    cv2.namedWindow(f'All digits dilated, kernel ({kernelWidth},{kernelHeight})', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'All digits dilated, kernel ({kernelWidth},{kernelHeight})', 1400, 840)  
    cv2.imshow(f'All digits dilated, kernel ({kernelWidth},{kernelHeight})', comparisonImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    comparisonImage[2*imgHeight:6*imgHeight, 0:10*imgWidth] = 0

    comparison_loop()

def compare_erode():
    cv_erode()
    kernel, kernelHeight, kernelWidth = create_kernel()
    for i in range(len(numberIndices)):
        comparisonImage[5*imgHeight:6*imgHeight, i*imgWidth:(i+1)*imgWidth] = erode(images[numberIndices[i]], kernel, kernelHeight, kernelWidth)
    
    text = "Custom Eroded Digits:"
    org = (imgWidth, 5*imgHeight - 2)
    cv2.putText(comparisonImage, text, org, font, font_scale, font_color, thickness)

    cv2.namedWindow(f'All digits eroded, kernel ({kernelWidth},{kernelHeight})', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'All digits eroded, kernel ({kernelWidth},{kernelHeight})', 1400, 840)  
    cv2.imshow(f'All digits eroded, kernel ({kernelWidth},{kernelHeight})', comparisonImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    comparisonImage[2*imgHeight:6*imgHeight, 0:10*imgWidth] = 0

    comparison_loop()

def compare_opening():
    cv_opening()
    kernel, kernelHeight, kernelWidth = create_kernel()
    for i in range(len(numberIndices)):
        comparisonImage[5*imgHeight:6*imgHeight, i*imgWidth:(i+1)*imgWidth] = opening(images[numberIndices[i]], kernel, kernelHeight, kernelWidth)
    
    text = "Custom Opened Digits:"
    org = (imgWidth, 5*imgHeight - 2)
    cv2.putText(comparisonImage, text, org, font, font_scale, font_color, thickness)

    cv2.namedWindow(f'All digits opened, kernel ({kernelWidth},{kernelHeight})', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'All digits opened, kernel ({kernelWidth},{kernelHeight})', 1400, 840)  
    cv2.imshow(f'All digits opened, kernel ({kernelWidth},{kernelHeight})', comparisonImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    comparisonImage[2*imgHeight:6*imgHeight, 0:10*imgWidth] = 0

    comparison_loop()

def compare_closing():
    cv_closing()
    kernel, kernelHeight, kernelWidth = create_kernel()
    for i in range(len(numberIndices)):
        comparisonImage[5*imgHeight:6*imgHeight, i*imgWidth:(i+1)*imgWidth] = closing(images[numberIndices[i]], kernel, kernelHeight, kernelWidth)
    
    text = "Custom Closed Digits:"
    org = (imgWidth, 5*imgHeight - 2)
    cv2.putText(comparisonImage, text, org, font, font_scale, font_color, thickness)

    cv2.namedWindow(f'All digits closed, kernel ({kernelWidth},{kernelHeight})', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'All digits closed, kernel ({kernelWidth},{kernelHeight})', 1400, 840)  
    cv2.imshow(f'All digits closed, kernel ({kernelWidth},{kernelHeight})', comparisonImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    comparisonImage[2*imgHeight:6*imgHeight, 0:10*imgWidth] = 0

    comparison_loop()

def comparison_loop():
    global comparing
    comparing = True
    print("_____________________________________________________")
    print("Compare openCV functions to custom functions. Choose options: \n<1> Dilate \n<2> Erode \n<3> Open \n<4> Close \n<5> Test Only OpenCV functions \n<6> Exit Program")
    choice = input()
    if choice == "1":
        compare_dilate()
    elif choice == "2":
        compare_erode()
    elif choice == "3":
        compare_opening()
    elif choice == "4":
        compare_closing()
    elif choice == "5":
        test_loop()
    elif choice == "6":
        exit()
    else:
        print(f"{choice} is not a valid option. \n")
        test_loop()   

def test_loop():
    global comparing
    comparing = False
    print("_____________________________________________________")
    print("Testing openCV functions. Choose options: \n<1> Dilate \n<2> Erode \n<3> Open \n<4> Close \n<5> Compare Custom functions performance \n<6> Exit Program")
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
        comparison_loop()
    elif choice == "6":
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
allNumbersImage = np.zeros((4*imgHeight, 10 * imgWidth), dtype=np.uint8)
comparisonImage = np.zeros((6*imgHeight, 10 * imgWidth), dtype=np.uint8)
for i in range(len(numberIndices)):
    allNumbersImage[imgHeight:2*imgHeight, i*imgWidth:(i+1)*imgWidth] = images[numberIndices[i]]
    comparisonImage[imgHeight:2*imgHeight, i*imgWidth:(i+1)*imgWidth] = images[numberIndices[i]]

text = "Original Digits:"
org = (imgWidth, imgHeight - 2)
cv2.putText(allNumbersImage, text, org, font, font_scale, font_color, thickness)
cv2.putText(comparisonImage, text, org, font, font_scale, font_color, thickness)


# cv2.imshow('One of each digit', allNumbersImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

test_loop()