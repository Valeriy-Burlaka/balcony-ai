import statistics

from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

import cv2
# import kagglehub

import numpy as np
import tensorflow as tf

from birds.lib.image_preprocessing import (
    normalize_for_tf,
    preprocess_image,
    read_image,
    # show_image,
    BGR_COLOR_RED
)
from birds.lib.coco_labels import COCO_LABELS, BIRD_CLASS, ALL_ANIMATE_CREATURES_CLASSES
from birds.lib.utils import timer


# IMAGE_PATH = "test-detection/clips-split-by-frames/frame00075.png"
# INPUT_IMAGE_FRAME_RANGE = range(65, 775)
# INPUT_IMAGES = [f"test-detection/clips-split-by-frames/frame{str(num).zfill(5)}.png" for num in INPUT_IMAGE_FRAME_RANGE]
# MODEL_VERSION = "d7"
# MODEL_PATH = kagglehub.model_download(f"tensorflow/efficientdet/tensorFlow2/{MODEL_VERSION}")
# MODEL_PATH = Path(f"~/.cache/kagglehub/models/tensorflow/efficientdet/tensorFlow2/{MODEL_VERSION}/1").expanduser()
INPUT_IMG_WIDTH = 1920
INPUT_IMG_HEIGHT = 1080
IMG_SIZE_FOR_DETECTOR = 640


class Box(NamedTuple):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

@dataclass
class ModelPerformanceStats:
    bird_detection_rate: float
    confidence_mean: float
    confidence_median: float
    confidence_std_dev: float
    other_creatures_detection_rate: float

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

class AnnotationResult(NamedTuple):
    image: np.ndarray
    num_birds_detected: int
    num_other_creatures_detected: int
    bird_confidence_scores: list[float]

def annotate_birds_and_other_animate_creatures(img, boxes, scores, classes, score_threshold=0.2) -> AnnotationResult:
    birds, other_creatures = 0, 0
    bird_confidence_scores = []

    for box, cls, score in zip(boxes, classes, scores):
        cls = int(cls)
        if cls in ALL_ANIMATE_CREATURES_CLASSES and score > score_threshold:
            if cls == BIRD_CLASS:
                birds += 1
                bird_confidence_scores.append(score)
            else:
                other_creatures += 1

            y_min, x_min, y_max, x_max = box
            box = denormalize_box_coordinates(Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))

            cv2.rectangle(img,
                          (box.x_min, box.y_min),
                          (box.x_max, box.y_max),
                          color=BGR_COLOR_RED,
                          thickness=1)

            label = COCO_LABELS[cls]
            cv2.putText(img,
                        f"{label}: {score:.2f}",
                        (box.x_min, box.y_min - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9,
                        color=(36,255,12),
                        thickness=2)
        else:
            continue

    return AnnotationResult(
        image=img,
        num_birds_detected=birds,
        num_other_creatures_detected=other_creatures,
        bird_confidence_scores=bird_confidence_scores)


def summarize_model_results(results: list[tuple[int, float, int]]) -> ModelPerformanceStats:
    bird_detections = [r[0] for r in results]
    confidence_scores = [r[1] for r in results]
    other_creaturs_detections = [r[2] for r in results]

    return ModelPerformanceStats(
        bird_detection_rate=statistics.mean(bird_detections),
        confidence_mean=statistics.mean(confidence_scores) if confidence_scores else 0,
        confidence_median=statistics.median(confidence_scores) if confidence_scores else 0,
        confidence_std_dev=statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
        other_creatures_rate=statistics.mean(other_creaturs_detections)
    )

# with timer("Loading the model"):
#     model = tf.saved_model.load(MODEL_PATH)

# output_dir = f"test-detection/model-outputs/efficientdet-{MODEL_VERSION}"
# if not Path(output_dir).exists():
#     Path(output_dir).mkdir(parents=True)



# TODO: Write csv with performance stats & bird/no-bird summary
# TODO: Replace "range of images" with input dataset dir

results = []

# input_image_frame_range = range(65, 775)
input_image_frame_range = range(65, 80)
input_images = [f"test-detection/clips-split-by-frames/frame{str(num).zfill(5)}.png" for num in input_image_frame_range]

for model_version in ["d0", "d1", "d2"]:
    print(f"Testing frames [{input_image_frame_range}] with '{model_version}'")
    with timer("Loading the model"):
        model_path = Path(f"~/.cache/kagglehub/models/tensorflow/efficientdet/tensorFlow2/{model_version}/1").expanduser()
        model = tf.saved_model.load(model_path)

    model_output_dir = f"test-detection/model-outputs/efficientdet-{model_version}"
    if not Path(model_output_dir).exists():
        Path(model_output_dir).mkdir(parents=True)

    for image_path in input_images:
        print(f"Now processing image '{image_path}'")

        img_name = Path(image_path).name
        orig_image = read_image(image_path)
        preprocessed = preprocess_image(orig_image, target_size=IMG_SIZE_FOR_DETECTOR)

        # cv2.imwrite(f"{output_dir}/preprocessed-{img_name}", preprocessed)

        normalized = normalize_for_tf(image=preprocessed)

        with timer("Object detection"):
            detector_output = model(normalized)

        boxes = detector_output["detection_boxes"][0].numpy()
        classes = detector_output["detection_classes"][0].numpy()
        scores = detector_output["detection_scores"][0].numpy()

        annotation_result = annotate_birds_and_other_animate_creatures(
            img=orig_image,
            boxes=boxes,
            classes=classes,
            scores=scores)

        results.append([model_version, img_name, annotation_result.num_birds_detected, annotation_result.num_other_creatures_detected])
        # TODO: Place outputs for detection failures into separate directories (i.e., no-bird, low-conf (<0.3, <0.51), cat/dog/other-animal)
        labeled_image = annotation_result.image

        cv2.imwrite(f"{model_output_dir}/labeled-{img_name}", labeled_image)

print("Model Version\tImage Name\tNum Birds Detected\tNum Other Creatures Detected")
for r in results:
    print(f"{r[0]}\t\t{r[1]}\t{r[2]}\t\t\t{r[3]}")
