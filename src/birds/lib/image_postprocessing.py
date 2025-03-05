from typing import NamedTuple

import cv2
import numpy as np

from birds.lib.coco_labels import ALL_ANIMATE_CREATURES_CLASSES, BIRD_CLASS, COCO_LABELS
from birds.lib.colors import BGR_COLOR_LIME, BGR_COLOR_RED


class Box(NamedTuple):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


def denormalize_box_coordinates(box: Box, input_img_width: int, input_img_height: int) -> Box:
    """
    Transform tensor coordinates (0...1) back to the coordinates on a 1920x1920 square image.
    """
    # Translate the Y coordinate values on the 1920x1920 square image to the values on the original
    # 1080x1920 image by subtracting the height of a single vertical bar, which we've added to
    # letterbox the original landscape image into a square.
    de_letterbox_value = (input_img_width - input_img_height) // 2

    x_min, x_max = int(box.x_min * input_img_width), int(box.x_max * input_img_width)
    y_min, y_max = int(box.y_min * input_img_width - de_letterbox_value), \
                    int(box.y_max * input_img_width - de_letterbox_value)

    return Box(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def annotate_image_with_selected_classes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    selected_classes=ALL_ANIMATE_CREATURES_CLASSES[:],
    score_threshold=0.2,
) -> np.ndarray:
    birds, other_creatures = 0, 0
    bird_confidence_scores = []

    input_image_height = image.shape[0]
    input_image_width = image.shape[1]
    print(f"Input image height: {input_image_height}; Width: {input_image_width}")

    for box, cls, score in zip(boxes, classes, scores):
        cls = int(cls)
        if cls in selected_classes and score > score_threshold:
            if cls == BIRD_CLASS:
                birds += 1
                bird_confidence_scores.append(score)
            else:
                other_creatures += 1

            y_min, x_min, y_max, x_max = box
            box = denormalize_box_coordinates(
                box=Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                input_img_width=input_image_width,
                input_img_height=input_image_height,
            )

            cv2.rectangle(
                image,
                (box.x_min, box.y_min),
                (box.x_max, box.y_max),
                color=BGR_COLOR_RED,
                thickness=1,
            )

            label = COCO_LABELS[cls]
            cv2.putText(
                image,
                f"{label}: {score:.2f}",
                (box.x_min, box.y_min - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                color=BGR_COLOR_LIME,
                thickness=2,
            )
        else:
            continue

    return image
