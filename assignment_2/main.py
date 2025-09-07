import os
import numpy as np
from pathlib import Path
import cv2

def padding(image, border_width=100):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width,border_width,border_width,border_width,
        cv2.BORDER_REFLECT
    )
    return padded_image

def crop(image,x_0,x_1,y_0,y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    return cropped_image

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y][x][c] = image[y][x][c]
    return emptyPictureArray

def grayscale(image):
    gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def hsv(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsvImage

def hue_shifted(image, emptyPictureArray, hue):
    hueShifted = (image.astype(np.int16) + int(hue)) % 256
    hueShifted = hueShifted.astype(np.uint8)
    np.copyto(emptyPictureArray, hueShifted)
    return emptyPictureArray

def smoothing(image):
    smoothedImage = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    return smoothedImage

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError('Rotation angle must be between 90 and 180 degrees')

def main():
    img = cv2.imread('lena-1.png',3)
    h,w = img.shape[:2]

    padded_image = padding(img, border_width=100)
    cropped_image = crop(img, 80, w-130, 80, h-130)
    resized_image = resize(img, 200, 200)

    emptyPictureArray = np.zeros((h, w, 3), dtype=np.uint8)
    copied_image = copy(img, emptyPictureArray)
    grayscale_image = grayscale(img)
    hsv_image = hsv(img)

    emptyHueArray = np.zeros((h, w, 3), dtype=np.uint8)
    hue_shifted_image = hue_shifted(img, emptyHueArray, hue= 50)
    smoothed_image = smoothing(img)

    rotated_90_image = rotation(img, 90)
    rotated_180_image = rotation(img, 180)

    # Save them to solution folder
    cv2.imwrite("solutions/padded_image.png", padded_image)
    cv2.imwrite("solutions/cropped_image.png", cropped_image)
    cv2.imwrite("solutions/resized_image.png", resized_image)
    cv2.imwrite("solutions/copied_image.png", copied_image)
    cv2.imwrite("solutions/grayscale_image.png", grayscale_image)
    cv2.imwrite("solutions/hsv_image.png", hsv_image)
    cv2.imwrite("solutions/hue_shifted_image.png", hue_shifted_image)
    cv2.imwrite("solutions/smoothing_image.png", smoothed_image)
    cv2.imwrite("solutions/rotated_90_image.png", rotated_90_image)
    cv2.imwrite("solutions/rotated_18_image.png", rotated_180_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()