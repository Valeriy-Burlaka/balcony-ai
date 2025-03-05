import statistics

from dataclasses import dataclass, field
from enum import auto, Enum
from pathlib import Path
from typing import NamedTuple

import cv2

import numpy as np
from rich.console import Console
from rich.table import Table

from birds.lib.image_preprocessing import (
    normalize_for_tf,
    preprocess_image,
    read_image,
)
from birds.lib.utils import timeit

from birds.lib.coco_labels import COCO_LABELS, BIRD_CLASS, ALL_ANIMATE_CREATURES_CLASSES
from birds.lib.colors import BGR_COLOR_RED

INPUT_IMG_WIDTH = 1920
INPUT_IMG_HEIGHT = 1080
# It *looks* like the effdet model is configured to work best with this image size
# (see https://github.com/google/automl/blob/master/efficientdet/hparams_config.py#L181).
# Some people on the Internet recommend to always use square images and to set both width
# and height of the input image to the multiples of 16, which sounds reasonable, and which
# I was already doing anyway.
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
class SingleDetectionResult:
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
        input_image_height = self.image_array.shape[0]
        input_image_width = self.image_array.shape[1]
        for box, cls, score in zip(self.boxes, self.classes, self.scores):
            cls = int(cls)
            if cls in ALL_ANIMATE_CREATURES_CLASSES and score > self.score_threshold:
                if cls == BIRD_CLASS:
                    self.num_birds_detected += 1
                    self.bird_confidence_scores.append(score)
                else:
                    self.num_other_creatures_detected += 1

                y_min, x_min, y_max, x_max = box
                box = denormalize_box_coordinates(
                    box=Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    input_img_width=input_image_width,
                    input_img_height=input_image_height,
                )

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
    def bird_confidence_mean(self) -> float:
        return statistics.mean(self.bird_confidence_scores) if self.bird_confidence_scores else .0

    @property
    def bird_confidence_max(self) -> float:
        return max(self.bird_confidence_scores) if self.bird_confidence_scores else .0

    @property
    def bird_confidence_min(self) -> float:
        return min(self.bird_confidence_scores) if self.bird_confidence_scores else .0

    def _assert_annotated(self):
        if not self.annotated:
            raise RuntimeWarning("Image is not annotated. Call .annotate_birds_and_other_animate_creatures() instance method first")

    @property
    def annotated_image(self) -> np.ndarray:
        self._assert_annotated()

        return self.image_array

    @property
    def result_categories(self) -> list[ResultCategory]:
        self._assert_annotated()

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

    def save(self, output_path: Path):
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)

        cv2.imwrite(output_path.as_posix(), self.annotated_image)

    def save_categorized_results(self, base_dir: str):
        self._assert_annotated()

        for category in self.result_categories:
            category_dir = SingleDetectionResult.result_category_to_dir_mapping()[category]
            p = Path(base_dir) / category_dir / self.image_fname

            self.save(p)

@dataclass
class ExperimentSummaryStats:
    model_version: str
    # Mean detection rate. Helpful for testing the uniform datasets where all frames in the datasets
    # are in the same category (i.e., either "bird" or "no bird").
    bird_detection_rate: float
    # A complementary metric for 'bird_detection_rate' metric. Helpful for testing the datasets where
    # all frames are known to have the same amount of birds.
    num_birds_detected_mean: float
    # A median calculates across the max confidence score of each frame
    bird_confidence_max_median: float
    # A median calculates across the min confidence score of each frame
    bird_confidence_min_median: float

    # confidence_std_dev: float
    # Capturing the detection rate for cats, dogs, cows, and other noise.
    other_creatures_detection_rate: float

    # Mean detection time for a model.
    processing_time_mean: float

    @classmethod
    def summarize_detection_results(cls, model_version: str, results: list[SingleDetectionResult]) -> "ExperimentSummaryStats":
        return cls(
            model_version=model_version,
            bird_detection_rate=statistics.mean([int(bool(r.num_birds_detected)) for r in results]),
            num_birds_detected_mean=statistics.mean([r.num_birds_detected for r in results]),
            bird_confidence_max_median=statistics.median([r.bird_confidence_max for r in results]),
            bird_confidence_min_median=statistics.median([r.bird_confidence_min for r in results]),
            other_creatures_detection_rate=statistics.mean([r.num_other_creatures_detected for r in results]),
            processing_time_mean=statistics.mean([r.t_spent for r in results])
        )

    @staticmethod
    def display_stats_table(stats_list: list["ExperimentSummaryStats"]):
        table = Table(title="Model Performance Summary Stats")

        # Add columns
        table.add_column("Metric", style="cyan", no_wrap=True)
        for stat in stats_list:
            table.add_column(f"{stat.model_version}", style="magenta")

        # Add rows
        metrics = [
            "Bird Detection Rate",
            "Num Birds Detected Mean",
            "Bird Confidence Max Median",
            "Bird Confidence Min Median",
            "Other Creatures Detection Rate",
            "Processing Time Mean"
        ]

        for metric in metrics:
            row = [metric]
            for stat in stats_list:
                value = getattr(stat, metric.lower().replace(" ", "_"))
                row.append(f"{value:.4f}")
            table.add_row(*row)

        # Print the table
        console = Console()
        console.print(table)

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

# def get_annotated_img_objects(img, boxes, scores, classes, score_threshold=0.25, label_map=None):
#     num_objects = 0

#     for box, cls, score in zip(boxes, classes, scores):
#         if score >= score_threshold:
#             y_min, x_min, y_max, x_max = box
#             box = denormalize_box_coordinates(Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))

#             cv2.rectangle(
#                 img,
#                 (box.x_min, box.y_min),
#                 (box.x_max, box.y_max),
#                 color=BGR_COLOR_RED,
#                 thickness=1,
#             )

#             if label_map:
#                 label = label_map[int(cls)]
#                 cv2.putText(
#                     img,
#                     f"{label}: {score:.2f}",
#                     (box.x_min, box.y_min - 10),
#                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=0.9,
#                     color=BGR_COLOR_LIME,
#                     thickness=2,
#                 )

#             num_objects += 1

#     return img, num_objects

def detect(model, image: Path) -> SingleDetectionResult:
    with timeit("Reading image") as t_spent:
        orig_image = read_image(image.as_posix())

    with timeit("Preprocessing image") as t_spent:
        preprocessed = preprocess_image(orig_image, target_size=IMG_SIZE_FOR_DETECTOR)

    with timeit("Normalizing image") as t_spent:
        normalized = normalize_for_tf(image=preprocessed)

    with timeit("Object detection") as t_spent:
        detector_output = model(normalized)

    boxes = detector_output["detection_boxes"][0].numpy()
    classes = detector_output["detection_classes"][0].numpy()
    scores = detector_output["detection_scores"][0].numpy()
    detection_result = SingleDetectionResult(
        image_fname=image.name,
        image_array=orig_image,
        boxes=boxes,
        classes=classes,
        scores=scores,
        score_threshold=0.2,
        t_spent=t_spent["seconds"])

    detection_result.annotate_birds_and_other_animate_creatures()

    return detection_result
