import cv2
import os
import numpy as np
from setuptools.sandbox import save_path


def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=1)
    sobuel_uint8 = cv2.convertScaleAbs(sobel)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "Solutions", "sobel_edge_detection.png")

    cv2.imwrite(save_path, sobuel_uint8)
    return sobuel_uint8

def canny_edge_detection(image,threshold_1=50, threshold_2=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(image, threshold_1, threshold_2)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "Solutions", "canny_edge_detection.png")

    cv2.imwrite(save_path, edges)
    return edges



def template_match(image, template, threshold=0.9):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template

    temp_height, temp_width = template_gray.shape[:2]
    img_height, img_width = image_gray.shape[:2]
    if temp_height > img_height or temp_width > img_width:
        raise ValueError("Template must be smaller or equal to image")

    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= threshold)

    out = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out = out.copy()
    for (x, y) in zip(xs, ys):
        cv2.rectangle(out, (x, y), (x + temp_width, y + temp_height), (0, 0, 255), 2)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "Solutions", "template_match.png")
    cv2.imwrite(save_path, out)
    return out

def resize_image(image, scale_factor: int, up_or_down:str):
    if up_or_down not in ["up", "down"]:
        raise ValueError("up_or_down must be 'up' or 'down'")

    out = image.copy()
    for _ in range(scale_factor):
        if up_or_down == "up":
            height, width = out.shape[:2]
            out = cv2.pyrUp(out, dstsize=(width*2, height*2))
        else:
            height, width = out.shape[:2]
            out = cv2.pyrDown(out, dstsize=(width//2, height//2))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "Solutions", f"resize_{up_or_down}_{scale_factor}.png")
    cv2.imwrite(save_path, out)
    return out

def main():
    img = cv2.imread('lambo.png')
    # Run sobel
    sobel_edge_detection(img)
    # Run canny
    canny_edge_detection(img)
    # Template match
    shapes =  cv2.imread('shapes-1.png')
    shape_template = cv2.imread('shapes_template.jpg')
    template_match(shapes, shape_template, threshold=0.9)
    # Resizing
    resize_image(img, scale_factor=1, up_or_down="down")
    resize_image(img, scale_factor=1, up_or_down="up")

    """
    sobel_img = sobel_edge_detection(img)
    cv2.imshow ('sobel', sobel_img)
    canny_img = canny_edge_detection(img)
    cv2.imshow('Canny', canny_img)
    """

    # Original image
    cv2.imshow('Original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()