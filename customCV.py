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

def erode(img, kernel=np.ones((3, 3), dtype=np.uint8), kernelHeight=3, kernelWidth=3):
    padX = kernelWidth // 2
    padY = kernelHeight // 2
    
    imgHeight, imgWidth = img.shape

    paddedImage = pad(img, kernelHeight, kernelWidth)
    erodedImage = np.zeros_like(img, dtype=np.uint8)

    for i in range(padY, imgHeight + padY):
        for j in range(padX, imgWidth + padX):
            roi = paddedImage[i-padY:i+padY+1, j-padX:j+padX+1]
            if kernelWidth % 2 == 0:
                roi = roi[:, :-1]
            if kernelHeight % 2 == 0:
                roi = roi[:-1, :]
            if np.all(roi[kernel == 1]):
                erodedImage[i-padY, j-padX] = 255

    return erodedImage


def dilate(img, kernel=np.ones((3, 3), dtype=np.uint8), kernelHeight=3, kernelWidth=3):
    padX = kernelWidth // 2
    padY = kernelHeight // 2

    imgHeight, imgWidth = img.shape
    
    paddedImage = pad(img, kernelHeight, kernelWidth)
    dilatedImage = np.zeros_like(img, dtype=np.uint8)

    for i in range(padY, imgHeight + padY):
        for j in range(padX, imgWidth + padX):
            roi = paddedImage[i-padY:i+padY+1, j-padX:j+padX+1]
            if kernelWidth % 2 == 0:
                roi = roi[:, :-1]
            if kernelHeight % 2 == 0:
                roi = roi[:-1, :]
            if np.any(roi[kernel > 0]):
                dilatedImage[i-padY, j-padX] = 255

    return dilatedImage

def opening(img, kernel=np.ones((3, 3), dtype=np.uint8), kernelHeight=3, kernelWidth=3):
    erodedImage = erode(img, kernel, kernelHeight, kernelWidth)
    openedImage = dilate(erodedImage, kernel, kernelHeight, kernelWidth)
    return openedImage

def closing(img, kernel=np.ones((3, 3), dtype=np.uint8), kernelHeight=3, kernelWidth=3):
    dilatedImage = dilate(img, kernel, kernelHeight, kernelWidth)
    closedImage = erode(dilatedImage, kernel, kernelHeight, kernelWidth)
    return closedImage

def binarize(img, threshold=127):
    binaryImage = (img > threshold).astype(np.uint8) * 255
    return binaryImage

def fill_holes(img):
    invertedImage = cv2.bitwise_not(img)
    
    numLabels, labels, stats, centroids = CCA(invertedImage)
    
    background = np.argmax(stats[0:, 4])  # +1 to adjust for skipping the first component
    holes = []
    
    # kernelHeight = 2
    # kernelWidth = 2
    # kernel = np.ones((kernelHeight, kernelWidth), np.uint8)

    for i in range(1, numLabels):
        if i == background:
            continue  # Skip background
        
        x, y, w, h, area = stats[i]
        centroidX, centroidY = centroids[i]
        
        holes.append((int(centroidY), int(centroidX)))

    fullFilledImage = np.zeros_like(img, dtype=np.uint8)

    for centroid in holes:
        filledImage = np.zeros_like(img, dtype=np.uint8)
        filledImage[centroid[0]][centroid[1]] = 255
        
        while True:
            dilation = dilate(filledImage)
            newFilledImage = np.bitwise_and(dilation, invertedImage)

            if np.array_equal(filledImage, newFilledImage):
                fullFilledImage = np.bitwise_or(fullFilledImage, filledImage)
                break
            filledImage = newFilledImage
        
        result = np.bitwise_or(fullFilledImage, img)
    
    result = np.bitwise_or(fullFilledImage, img)
    
    return result

def find_holes(img):
    filledImage = fill_holes(img)
    return filledImage - img

def CCA(img, connectivity=8):
    imgHeight, imgWidth = img.shape

    labels = []
    equivalence = {}
    blobs = np.zeros((imgHeight+2, imgWidth+2),  dtype=np.uint8)
    for y in range(1, imgHeight+1):
        for x in range(1, imgWidth+1):
            if img[y-1][x-1] == 0:
                continue
            
            roi = blobs[y-1:y+1, x-1:x+1]
            if connectivity == 4:
                if roi[0][1] != 0 or roi[1][0] != 0 or roi[1][2] != 0 or roi[2][1] != 0:
                    values = [roi[0][1], roi[1][0], roi[1][2], roi[2][1]]
                    if 0 in values:
                        values.np.remove(0)
                    minVal = min(values)

                    for val in values:
                        if minVal != val:
                            if val in equivalence:
                                equivalence[val].append(minVal)
                            else:
                                equivalence[val] = [minVal]

                    blobs[y][x] = minVal
                else:
                    if not labels:
                        blobs[y][x] = 1
                        labels.append(1)
                    else:
                        blobs[y][x] = max(labels) + 1
                        labels.append(max(labels) + 1)
            else:
                if np.any(roi != 0):
                    values = roi[roi != 0]

                    minVal = min(values)

                    for val in values:
                        if minVal != val:
                            if val in equivalence:
                                equivalence[val].append(minVal)
                            else:
                                equivalence[val] = [minVal]

                    blobs[y][x] = minVal
                else:
                    if not labels:
                        blobs[y][x] = 1
                        labels.append(1)
                    else:
                        blobs[y][x] = max(labels) + 1
                        labels.append(max(labels) + 1)

    blobs = blobs[1:-1, 1:-1]

    for y in range(imgHeight - 1):
        for x in range(imgWidth - 1):
            if blobs[y][x] > 0:
                currentLabel = blobs[y][x]
                while currentLabel in equivalence:
                    currentLabel = min(equivalence[currentLabel])
                blobs[y][x] = currentLabel

    labels = np.unique(blobs)
    labels = labels[labels != 0]
    numLabels = len(labels)

    centroids = []
    stats = []

    for i in range(numLabels):
        blob = (blobs == labels[i])

        # Find the coordinates of all pixels with the current label
        yCoords, xCoords = np.nonzero(blob)
        ## x y width height area
        stats.append([min(xCoords), min(yCoords), (max(xCoords) - min(xCoords)), (max(yCoords) - min(yCoords)), np.count_nonzero(blob)])

        # Compute the centroid
        if len(yCoords) > 0:  # Avoid division by zero
            centroid_x = int(np.mean(xCoords))
            centroid_y = int(np.mean(yCoords))
            centroids.append([centroid_x, centroid_y])

    return numLabels, labels, np.array(stats), centroids


def bounding_box(img):
    foregroundCoords = np.column_stack(np.where(img > 0))

    if foregroundCoords.size == 0:
        return None

    minY, minX = np.min(foregroundCoords, axis=0)
    maxY, maxX = np.max(foregroundCoords, axis=0)

    return minX, minY, maxX, maxY

## Uses Bresenham line drawing algorithm
def draw_line(img, x1, y1, x2, y2, colour=255, thickness=1):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    #step directions
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    img[y1][x1] = colour

    # Horizontal
    if dx > dy:
        err = dx // 2
        while x1 != x2:
            err -= dy
            if err < 0:
                y1 += sy
                err += dx
            x1 += sx
            set_pixel(img, x1, y1, colour, thickness)
    # Vertical
    else:
        err = dy // 2
        while y1 != y2:
            err -= dx
            if err < 0:
                x1 += sx
                err += dy
            y1 += sy
            set_pixel(img, x1, y1, colour, thickness)

    return img
  
def set_pixel(img, x, y, colour, thickness):
    if thickness == 1:
        img[y][x] = colour
        return
    for i in range(-thickness//2, thickness//2):
        for j in range(-thickness//2, thickness//2):
            if 0 <= x + i < img.shape[1] and 0 <= y + j < img.shape[0]:
                img[y + j][x + i] = colour


def draw_rect(img, x1, y1, x2, y2, colour=255, thickness=1):
    imgRect = np.copy(img)
    lines = [[x1, y1, x2, y1], #top
             [x1, y1, x1, y2], #left
             [x1, y2, x2, y2], #bottom
             [x2, y1, x2, y2]] #right
    
    for line in lines:
        imgRect = draw_line(imgRect, line[0], line[1], line[2], line[3], colour, thickness)

    return imgRect

def find_leftmost_high_pixel(img):
    height, width = img.shape

    for x in range(width):
        column = img[:, x]
        if np.any(column > 0): 
            y = np.argmax(column > 0)
            return x, y

    return None  # Return None if no foreground pixel is found

def find_leftmost_low_pixel(img):
    height, width = img.shape

    for x in range(width):
        column = img[:, x]
        if np.any(column > 0): 
            y = np.argmax(column[::-1] > 0)
            y = height - 1 - y
            return x, y

    return None  # Return None if no foreground pixel is found


def find_rightmost_high_pixel(img):
    height, width = img.shape

    for x in range(width - 1, -1, -1):
        column = img[:, x]
        if np.any(column > 0): 
            y = np.argmax(column > 0)
            return x, y

    return None  # Return None if no foreground pixel is found

def find_rightmost_low_pixel(img):
    height, width = img.shape

    for x in range(width - 1, -1, -1):
        column = img[:, x]
        if np.any(column > 0): 
            y = np.argmax(column[::-1] > 0)
            y = height - 1 - y
            return x, y

    return None  # Return None if no foreground pixel is found

def find_topmost_left_pixel(img):
    height, width = img.shape

    for y in range(height):
        row = img[y, :]
        if np.any(row > 0): 
            x = np.argmax(row > 0)
            return x, y

    return None  # Return None if no foreground pixel is found

def find_topmost_right_pixel(img):
    height, width = img.shape

    for y in range(height):
        row = img[y, :]
        if np.any(row > 0): 
            x = np.argmax(row[::-1] > 0)
            x = width - 1 - x
            return x, y

    return None  # Return None if no foreground pixel is found

# Returns 0 is stem is below blob, 1 if above, None is otherwise
def find_stem(img):
    holes = find_holes(img)
    _, _, _, centroids = CCA(holes)
    if not len(centroids) == 0:
        y = centroids[0][1]

        dilatedHole = dilate(holes)
        stemImg = img - dilatedHole
        stemImg = dilate(stemImg)
        _, _, _, centroids = CCA(stemImg)
        y2 = centroids[0][1]

        if y > y2:
            return 0
        
        return 1
    
    return None


