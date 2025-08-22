import numpy as np
import cv2
import os

def print_image_information(image):

    height, width, channels = image.shape
    print(".::Image shape details::.")
    print("Height:",height)
    print("Width:",width)
    print("Channels:",channels)
    print("Size:", image.size)
    print("Data type:", image.dtype)

def save_cam_information():

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Fetch cam info
    cam_fps = cam.get(cv2.CAP_PROP_FPS)
    cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # close cam
    cam.release()

    save_dir = "solutions"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "camera_outputs.txt")

    with open(file_path, "w") as f:
        f.write(f"fps: {cam_fps}\n")
        f.write(f"width: {cam_width}\n")
        f.write(f"height: {cam_height}\n")

    print(f"Camera outputs saved in {file_path}")


def main():
    img = cv2.imread('lena-1.png',1) # Opening with colors.
    print_image_information(img)
    save_cam_information()


if __name__ == "__main__":
    main()