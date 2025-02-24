import cv2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


# https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/detect.py

def main():
    # Initialize the object detection model
    model = "efficientdet_lite0.tflite"
    enable_edgetpu = True
    num_threads = 4

    base_options = core.BaseOptions(
        file_name=model,
        use_coral=enable_edgetpu,
        num_threads=num_threads,
    )
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)

    detector = vision.ObjectDetector.create_from_options(options)
    print(detector)

    # Classification
    # # Imports
    # from tflite_support.task import vision
    # from tflite_support.task import core

    # # Initialize options and turn on Coral Edge TPU delegation.
    # base_options = core.BaseOptions(file_name=model_path, use_coral=True)
    # options = vision.ImageClassifierOptions(base_options=base_options)

    # # Create ImageClassifier from options.
    # classifier = vision.ImageClassifier.create_from_options(options)

    # # Run inference on Coral Edge TPU.
    # image = vision.TensorImage.create_from_file(image_path)
    # classification_result = classifier.classify(image)


if __name__ == "__main__":
    main()
    # # Load an image
    # image_path = "test.jpg"
    # image = cv2.imread(image_path)

    # # Run inference
    # results = detector.process(image)

    # # Print the detection results
    # for result in results:
    #     print(result)
    #     print("\n")
