import os

import cv2
import numpy as np
import tensorflow as tf

print(f"Process PID: {os.getpid()}")

img_path = "test-detection/clips-split-by-frames/frame00075.png"
model_path = "/Users/val/.cache/kagglehub/models/tensorflow/efficientdet/tensorFlow2/d1/1"

# std_height, std_width = 640, 640
target_size = 640
gray_fill = (128, 128, 128)

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
    full_square_img = np.full((target_size, target_size, 3), gray_fill, dtype=np.uint8)
    vertical_offset = (target_size - img_height) // 2

    full_square_img[vertical_offset:vertical_offset+img_height, 0:img_width] = image[0:img_height, 0:img_width]

    return full_square_img

def downsample_image(image: np.ndarray, target_size: int) -> np.ndarray:
    img_height, img_width = image.shape[:2]
    if img_height != img_width:
        raise ValueError("Attempting to downsample a non-square image")

    return cv2.resize(image, (target_size, target_size), cv2.INTER_LANCZOS4)

def normalize(image: np.ndarray) -> np.ndarray:
    return image / 255.0

def preprocess_image(img_path: str):
    img = cv2.imread(img_path)
    # Convert the image from BGR to RGB as required by the TFLite model.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = letterbox_landscape_image_to_square(img)

    img = downsample_image(img, target_size=target_size)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def show_image(image: np.ndarray):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = preprocess_image(img_path=img_path)
cv2.imwrite("./preprocessed.png", img)

normalized = normalize(img)
# The operation tf.expand_dims(sample_image_t, axis=0) in TF adds a new dimension to the tensor T at the specified (0) axis.
# If the `normalized` image originally had a shape of (640, 640, 3), after applying tf.expand_dims(sample_image_t, axis=0),
# its new shape becomes (1, 640, 640, 3), effectively turning it into a "batch" of one image.
#  (Read more details here: https://claude.ai/chat/3805017f-9196-4989-9c66-0d3309a4f4df)
normalized = tf.expand_dims(normalized, axis=0)

print("\nTensor:\n", type(normalized), normalized.dtype, normalized.shape)

# model = tf.saved_model.load(model_path)
