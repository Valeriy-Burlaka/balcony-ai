import os

import cv2
import numpy as np

from birds.lib.colors import COLOR_GRAY

print(f"Process PID: {os.getpid()}")


def split_1080p_image_into_6_tiles(image: np.ndarray):
    # the initial idea was to have a specialized function to convert 1080p (1920x1080) videos to
    # a six 640x640 tiles. Instead of a creating a "generic" function that would handle all input image
    # sizes, handle just one, real use-case I have (phone camera). The algorithm would simply copy
    # the 1st horizontal row of 3 tiles as is, and handle the 2nd row using gray-padded tiles.
    # However, later I realized that the tiling approach will not cut it, as there could be numerous
    # edge cases with an object split between different tiles (4 in the worst-case scenario). Some
    # techniques like processing oerlapping tiles or re-constructing the object exist but all of them
    # heavily increase either code or computational complexity.
    pass

def letterbox_landscape_image_to_square(image: np.ndarray) -> np.ndarray:
    img_height, img_width = image.shape[:2]
    target_size = img_width
    full_square_img = np.full((target_size, target_size, 3), COLOR_GRAY, dtype=np.uint8)
    vertical_offset = (target_size - img_height) // 2

    full_square_img[vertical_offset:vertical_offset+img_height, 0:img_width] = image[0:img_height, 0:img_width]

    return full_square_img

def downsample_image(image: np.ndarray, target_size: int) -> np.ndarray:
    img_height, img_width = image.shape[:2]

    if img_height == img_width == target_size:
        return image

    if img_height != img_width:
        raise ValueError("Attempting to downsample a non-square image")

    return cv2.resize(image, (target_size, target_size), cv2.INTER_LANCZOS4)

def read_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)

def preprocess_image(image: np.ndarray, target_size: int) -> np.ndarray:
    # Convert the image from BGR to RGB as required by the TFLite model. # [26/11/24, Val]: Wat? Then why convert back before returning?
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = letterbox_landscape_image_to_square(image)

    image = downsample_image(image, target_size=target_size)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def normalize_for_tf(image: np.ndarray) -> np.ndarray:
    # image = image / 255.0  # not required by efficientdet d2, although I don't see a Rescaling layer in it either :shrug:

    # The operation np.expand_dims(sample_image_t, axis=0) adds a new dimension to the tensor T at the specified (0) axis.
    # If the `normalized` image originally had a shape of (640, 640, 3), after applying tf.expand_dims(sample_image_t, axis=0),
    # its new shape becomes (1, 640, 640, 3), effectively turning it into a "batch" of one image.
    #  (Read more details here: https://claude.ai/chat/3805017f-9196-4989-9c66-0d3309a4f4df)
    image = np.expand_dims(image, axis=0)

    return image

def show_image(image: np.ndarray):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
