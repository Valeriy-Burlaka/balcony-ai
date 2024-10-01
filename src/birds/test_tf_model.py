import time

from typing import NamedTuple
from pathlib import Path

import cv2
# import kagglehub
import tensorflow as tf

from birds.lib.image_preprocessing import (
    normalize_for_tf,
    preprocess_image,
    read_image,
    show_image,
    BGR_COLOR_RED
)
from birds.lib.coco_labels import COCO_LABELS
from birds.lib.utils import timer


IMAGE_PATH = "test-detection/clips-split-by-frames/frame00075.png"
# model_path = kagglehub.model_download("tensorflow/efficientdet/tensorFlow2/d1")
MODEL_PATH = Path("~/.cache/kagglehub/models/tensorflow/efficientdet/tensorFlow2/d1/1").expanduser()
INPUT_IMG_WIDTH = 1920
INPUT_IMG_HEIGHT = 1080
IMG_SIZE_FOR_DETECTOR = 640


class Box(NamedTuple):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

def denormalize_box_coordinates(box: Box) -> Box:
    # scale_back_coeff = INPUT_IMG_WIDTH / IMG_SIZE_FOR_DETECTOR
    # Transform tensor coordinates (0...1) back to the coordinates on a 1920x1920 square image.
    scale_back_coeff = INPUT_IMG_WIDTH
    # Translate the Y coordinate values on the 1920x1920 square image to the values on the original
    # 1080x1920 image by subtracting the height of a single vertical bar, which we've added to
    # letterbox the original landscape image into a square.
    de_letterbox_value = (INPUT_IMG_WIDTH - INPUT_IMG_HEIGHT) // 2

    x_min, x_max = int(box.x_min * scale_back_coeff), int(box.x_max * scale_back_coeff)
    y_min, y_max = int(box.y_min * scale_back_coeff - de_letterbox_value), \
                    int(box.y_max * scale_back_coeff - de_letterbox_value)

    return Box(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

def get_annotated_img_objects(img, boxes, scores, classes, score_threshold=0.25, label_map=None):
    num_objects = 0

    for box, cls, score in zip(boxes, classes, scores):
        if score >= score_threshold:
            y_min, x_min, y_max, x_max = box
            box = denormalize_box_coordinates(Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))

            cv2.rectangle(img,
                          (box.x_min, box.y_min),
                          (box.x_max, box.y_max),
                          color=BGR_COLOR_RED,
                          thickness=1)

            if label_map:
                label = label_map[int(cls)]
                cv2.putText(img,
                            f"{label}: {score:.2f}",
                            (box.x_min, box.y_min - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9,
                            color=(36,255,12),
                            thickness=2)

            num_objects += 1

    return img, num_objects


orig_image = read_image(IMAGE_PATH)
cv2.imwrite("./original.png", orig_image)

with timer("Loading the model"):
    model = tf.saved_model.load(MODEL_PATH)

with timer("Preprocessing the image"):
    preprocessed = preprocess_image(orig_image, target_size=IMG_SIZE_FOR_DETECTOR)

cv2.imwrite("./preprocessed.png", preprocessed)

normalized = normalize_for_tf(image=preprocessed)

with timer("Object detection"):
    detector_output = model(normalized)

boxes = detector_output["detection_boxes"][0].numpy()
classes = detector_output["detection_classes"][0].numpy()
scores = detector_output["detection_scores"][0].numpy()

labeled_image, num_detected_objects = get_annotated_img_objects(
    img=orig_image,
    boxes=boxes,
    scores=scores,
    classes=classes,
    label_map=COCO_LABELS,
)
cv2.imwrite("./original_labeled.png", labeled_image)

# Sandbox: process a single, top-score detection and draw its detection box over both
# preprocessed (640x640) and original (1920x1080) images.

# print("First box: ", boxes[0], boxes[0].shape, type(boxes[0]))
# print("First class: ", classes[0])
# print("First score: ", scores[0])

# y_min, x_min, y_max, x_max = boxes[0]

# box = Box(
#     x_min=int(x_min * IMG_SIZE_FOR_DETECTOR),
#     y_min=int(y_min * IMG_SIZE_FOR_DETECTOR),
#     x_max=int(x_max * IMG_SIZE_FOR_DETECTOR),
#     y_max=int(y_max * IMG_SIZE_FOR_DETECTOR))
# # print(f"Box coordinates for normalized (640x640) image: (x_min={x_min}; y_min={y_min}), (x_max={x_max}; y_max={y_max})", )
# print(f"Box coordinates for normalized (640x640) image: {box}", )
# cv2.rectangle(preprocessed,
#               (box.x_min, box.y_min),
#               (box.x_max, box.y_max),
#               color=BGR_COLOR_RED,
#               thickness=1)
# cv2.imwrite("./preprocessed_overlayed.png", preprocessed)

# box = denormalize_box_coordinates(Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))
# print("Box coordinates for denormalized (1920x1080) image: ", box)

# cv2.rectangle(orig_image,
#               (box.x_min, box.y_min),
#               (box.x_max, box.y_max),
#               color=BGR_COLOR_RED,
#               thickness=1)
# cv2.imwrite("./original_overlayed.png", orig_image)
