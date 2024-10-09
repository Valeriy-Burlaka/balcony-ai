import statistics

from dataclasses import dataclass, field
from enum import auto, Enum
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
from birds.lib.logger import get_logger
from birds.lib.utils import timeit


logger = get_logger("test_tf_pipeline", verbosity=2)


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

class ResultCategory(Enum):
    NO_BIRDS = auto()
    SINGLE_BIRD_HIGH_CONFIDENCE = auto()
    SINGLE_BIRD_LOW_CONFIDENCE = auto()
    MANY_BIRDS_HIGH_CONFIDENCE = auto()
    MANY_BIRDS_LOW_CONFIDENCE = auto()
    OTHER_ANIMALS = auto()

@dataclass
class DetectionResult:
    model_version: str
    image_fname: str
    image_array: np.ndarray
    t_spent: float
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    score_threshold: float

    annotated: bool = False
    num_birds_detected: int = 0
    num_other_creatures_detected: int = 0
    bird_confidence_scores: list[float] = field(default_factory=list)

    def annotate_birds_and_other_animate_creatures(self) -> np.ndarray:
        for box, cls, score in zip(self.boxes, self.classes, self.scores):
            cls = int(cls)
            if cls in ALL_ANIMATE_CREATURES_CLASSES and score > self.score_threshold:
                if cls == BIRD_CLASS:
                    self.num_birds_detected += 1
                    self.bird_confidence_scores.append(score)
                else:
                    self.num_other_creatures_detected += 1

                y_min, x_min, y_max, x_max = box
                box = denormalize_box_coordinates(Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))

                cv2.rectangle(self.image_array,
                            (box.x_min, box.y_min),
                            (box.x_max, box.y_max),
                            color=BGR_COLOR_RED,
                            thickness=1)

                label = COCO_LABELS[cls]
                cv2.putText(self.image_array,
                            f"{label}: {score:.2f}",
                            (box.x_min, box.y_min - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9,
                            color=(36,255,12),
                            thickness=2)
            else:
                continue

        self.annotated = True

        return self.image_array

    @property
    def result_categories(self) -> list[ResultCategory]:
        if not self.annotated:
            raise RuntimeWarning("Image is not annotated. Call .annotate_birds_and_other_animate_creatures() instance method first")

        categories = []

        match self.num_birds_detected:
            case 0:
                categories.append(ResultCategory.NO_BIRDS)
            case 1:
                rc = ResultCategory.SINGLE_BIRD_HIGH_CONFIDENCE \
                    if self.bird_confidence_scores[0] > 0.5 \
                    else ResultCategory.SINGLE_BIRD_LOW_CONFIDENCE
                categories.append(rc)
            case _:
                rc = ResultCategory.MANY_BIRDS_HIGH_CONFIDENCE \
                    if statistics.mean(self.bird_confidence_scores) > 0.5 \
                    else ResultCategory.MANY_BIRDS_LOW_CONFIDENCE
                categories.append(rc)

        if self.num_other_creatures_detected > 0:
            categories.append(ResultCategory.OTHER_ANIMALS)

        return categories

    @staticmethod
    def result_category_to_dir_mapping():
        return {
            ResultCategory.NO_BIRDS: "no_birds",
            ResultCategory.SINGLE_BIRD_HIGH_CONFIDENCE: "single_bird_high_conf",
            ResultCategory.SINGLE_BIRD_LOW_CONFIDENCE: "single_bird_low_conf",
            ResultCategory.MANY_BIRDS_HIGH_CONFIDENCE: "multiple_birds_high_conf",
            ResultCategory.MANY_BIRDS_LOW_CONFIDENCE: "multiple_birds_low_conf",
            ResultCategory.OTHER_ANIMALS: "others"
        }

    def save(self, base_dir: str):
        # if not self.annotated:
        #     logger.warning("Image is not annotated yet. Saving original image.")

        for category in self.result_categories:
            category_dir = DetectionResult.result_category_to_dir_mapping()[category]
            p = Path(base_dir) / category_dir
            if not p.exists():
                p.mkdir()

            p = p / self.image_fname

            cv2.imwrite(f"{str(p)}", self.image_array)

# @dataclass
class ModelPerformanceStats:
    # bird_detection_rate: float
    # confidence_mean: float
    # confidence_median: float
    # confidence_std_dev: float
    # other_creatures_detection_rate: float

    def __init__(self, model_version: str, results: list[DetectionResult]):
        self.version = model_version
        self.results = results

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

def annotate_birds_and_other_animate_creatures(img, boxes, scores, classes, score_threshold=0.2) -> DetectionResult:
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

    return DetectionResult(
        image_array=img,
        num_birds_detected=birds,
        num_other_creatures_detected=other_creatures,
        bird_confidence_scores=bird_confidence_scores)

# with timer("Loading the model"):
#     model = tf.saved_model.load(MODEL_PATH)

# output_dir = f"test-detection/model-outputs/efficientdet-{MODEL_VERSION}"
# if not Path(output_dir).exists():
#     Path(output_dir).mkdir(parents=True)



# TODO: Write csv with performance stats & bird/no-bird summary
# TODO: Replace "range of images" with input dataset dir

results = {}

# input_image_frame_range = range(65, 775)
input_image_frame_range = range(75, 80)
# input_image_frame_range = range(75, 76)
input_images = [f"test-detection/clips-split-by-frames/frame{str(num).zfill(5)}.png" for num in input_image_frame_range]

# for model_version in ["d0", "d1", "d2"]:
for model_version in ["d0"]:
    print(f"Testing frames [{input_image_frame_range}] with '{model_version}'")
    with timeit("Loading the model"):
        model_path = Path(f"~/.cache/kagglehub/models/tensorflow/efficientdet/tensorFlow2/{model_version}/1").expanduser()
        model = tf.saved_model.load(model_path)

    for image_path in input_images:
        print(f"Now processing image '{image_path}'")

        image_fname = Path(image_path).name
        orig_image = read_image(image_path)
        preprocessed = preprocess_image(orig_image, target_size=IMG_SIZE_FOR_DETECTOR)

        # cv2.imwrite(f"{output_dir}/preprocessed-{image_name}", preprocessed)

        normalized = normalize_for_tf(image=preprocessed)

        with timeit("Object detection") as t_spent:
            detector_output = model(normalized)
        t_spent_seconds = t_spent["seconds"]

        boxes = detector_output["detection_boxes"][0].numpy()
        classes = detector_output["detection_classes"][0].numpy()
        print("Type classes:", type(boxes))
        scores = detector_output["detection_scores"][0].numpy()
        print("Type scores:", type(boxes))

        detection_result = DetectionResult(
            model_version=model_version,
            image_fname=image_fname,
            image_array=orig_image,
            t_spent=t_spent_seconds,
            boxes=boxes,
            scores=scores,
            classes=classes,
            score_threshold=0.2)
        detection_result.annotate_birds_and_other_animate_creatures()

        results.setdefault(model_version, []).append(detection_result)

for model_version, results in results.items():
    model_output_dir = Path(f"test-detection/model-outputs/efficientdet-{model_version}")
    if not model_output_dir.exists():
        model_output_dir.mkdir(parents=True)

    logger.info("Saving labeled images to the file system")
    for r in results:
        r.save(base_dir=str(model_output_dir))


# print("Model Version\tImage Name\tNum Birds Detected\tNum Other Creatures Detected")
# for r in results:
#     print(f"{r[0]}\t\t{r[1]}\t{r[2]}\t\t\t{r[3]}")
