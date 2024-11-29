import cv2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


# https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/detect.py

# Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)
