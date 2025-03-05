from pathlib import Path
import time

import cv2

import numpy as np
import streamlit as st

from PIL import Image
from streamlit_cropper import st_cropper
from tflite_runtime.interpreter import Interpreter, load_delegate

# from birds.lib.image_detection import (
#     annotate_image_with_selected_classes,
#     IMG_SIZE_FOR_DETECTOR,
# )
# from birds.lib.image_preprocessing import normalize_for_tf, preprocess_image
from birds.lib.logger import get_logger, update_app_verbosity_level


# logger = get_logger("web_app", verbosity=2)
# update_app_verbosity_level(verbosity=2)


def get_models():
    models_dir = Path.cwd() / "models" # todo: __init__.py in the `models` dir setting path & import
    models = [p.name for p in models_dir.glob("*.tflite")]

    return models

@st.cache_resource
def load_model(model_name: str):
    t1 = time.monotonic()
    model_path = Path.cwd() / "models" / model_name
    if "edgetpu" in model_path.name:
        interpreter = Interpreter(
            model_path=model_path.as_posix(),
            experimental_delegates=[load_delegate("libedgetpu.so.1")])
    else:
        interpreter = Interpreter(model_path=model_path.as_posix())

    interpreter.allocate_tensors()
    t_spent = time.monotonic() - t1

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    required_size = (input_shape[1], input_shape[2])

    return interpreter, required_size, t_spent


def attach_model_picker(app):
    available_models = get_models()
    selected_model = st.sidebar.selectbox(
        "Select model",
        available_models,
        index=available_models.index("efficientdet_lite0.tflite"))

    return selected_model


def start_app():
    st.set_page_config(
        layout="wide",
        page_title="Balcony life",
        page_icon="🦉"
    )
    st.header("Turn any Python script into a compelling web app 🐦🐦‍⬛🐓🦉🦅")
    st.button("Rerun")

    left_column, right_column = st.columns(2)

    selected_model = attach_model_picker(st)
    interpreter, required_size, t_spent = load_model(selected_model)

    st.sidebar.write(f"Model loaded in {t_spent} seconds")
    st.sidebar.write(f"Model input details: {interpreter.get_input_details()}")

    uploaded_image = st.sidebar.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg"],
    )
    cropped_image = None

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        with left_column:
            st.write("Double click to save crop")

            cropped_image = st_cropper(
                img_file=img,
                aspect_ratio=(1, 1),
                realtime_update=False,
                stroke_width=2,
            )
            # st.image(uploaded_image, caption="Uploaded image", use_column_width=True)
        with left_column:
            _ = cropped_image.thumbnail((required_size[0], required_size[1]))
            st.image(cropped_image, caption="Cropped image")


    # print("Cropped image: ", cropped_image)


    if uploaded_image is not None:
        # Resize and normalize image
        t_convert = time.monotonic()
        image = np.asarray(bytearray(uploaded_image.getvalue()))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, required_size)
        input_data = np.expand_dims(image_resized, axis=0)
        st.sidebar.write(f"Image pre-processed in {time.monotonic() - t_convert} seconds")

        # Run inference
        t_inference = time.monotonic()
        interpreter.set_tensor(0, input_data)
        interpreter.invoke()
        st.sidebar.write("Time spent for inference: ", time.monotonic() - t_inference)

        # Get results
        output_details = interpreter.get_output_details()
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num_detections = interpreter.get_tensor(output_details[3]['index'])
        st.sidebar.write(f"Number of detections: {num_detections}")
        st.sidebar.write(f"Classes: {classes}")
        st.sidebar.write(f"Scores: {scores}")
        st.sidebar.write(f"Boxes: {boxes}")

    #     original_image_as_ndarray = cv2.imdecode(uploaded_image_data, cv2.IMREAD_COLOR)
    #     prepocessed = preprocess_image(original_image_as_ndarray, target_size=IMG_SIZE_FOR_DETECTOR)
    #     normalized = normalize_for_tf(image=prepocessed)
    #     detector_output = model(normalized)
    #     boxes = detector_output["detection_boxes"][0].numpy()
    #     classes = detector_output["detection_classes"][0].numpy()
    #     scores = detector_output["detection_scores"][0].numpy()

    #     annotated = annotate_image_with_selected_classes(
    #         image=original_image_as_ndarray,
    #         boxes=boxes,
    #         classes=classes,
    #         scores=scores,
    #     )
    #     annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    #     with right_column:
    #         st.image(annotated, caption="To hell and back")

    # if cropped_image is not None:
    #     cropped_image_as_ndarray = np.array(cropped_image, dtype=np.uint8)
    #     print(cropped_image_as_ndarray.shape)

    #     # prepocessed = preprocess_image(cropped_image_as_ndarray, target_size=IMG_SIZE_FOR_DETECTOR)
    #     normalized = normalize_for_tf(image=cropped_image_as_ndarray)
    #     detector_output = model(normalized)
    #     boxes = detector_output["detection_boxes"][0].numpy()
    #     classes = detector_output["detection_classes"][0].numpy()
    #     scores = detector_output["detection_scores"][0].numpy()

    #     print("Classes", classes)
    #     print("Scores", scores)
    #     print("Boxes", boxes)


    #     annotated = annotate_image_with_selected_classes(
    #         image=cropped_image_as_ndarray,
    #         boxes=boxes,
    #         classes=classes,
    #         scores=scores,
    #     )
    #     # annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    #     with right_column:
    #         st.image(annotated, caption="To hell and back (twice)")


if __name__ == "__main__":
    start_app()

# t2 = time.monotonic()
#     image = cv2.imread(test_img)
#     print("Time spent for reading the image: ", time.monotonic() - t2)
#     input_shape = input_details[0]['shape']

#     #required_size = (input_shape[1], input_shape[2])  # Usually 320x320 for EfficientDet-Lite0
#     required_size = (640, 640)  # Usually 320x320 for EfficientDet-Lite0

#     # Resize and normalize image
#     t3 = time.monotonic()
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_resized = cv2.resize(image_rgb, required_size)
#     input_data = np.expand_dims(image_resized, axis=0)
#     print("Time spent to prepare the image: ", time.monotonic() - t3)

#     # IMPORTANT: Do NOT normalize to [-1, 1]. The model expects uint8 and handles normalization internally.
#     #input_data = (input_data.astype(np.float32) / 127.5) - 1  # Normalize to [-1, 1]

#     # Run inference
#     t4 = time.monotonic()
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()
#     print("Time spent for inference: ", time.monotonic() - t4)

#     # Get results
#     boxes = interpreter.get_tensor(output_details[0]['index'])
#     classes = interpreter.get_tensor(output_details[1]['index'])
#     print(classes)
#     scores = interpreter.get_tensor(output_details[2]['index'])
#     print(scores)
#     num_detections = interpreter.get_tensor(output_details[3]['index'])

#     # Process results (example: print detections above 0.5 confidence)
#     for i in range(int(num_detections[0])):
#         if scores[0][i] > 0.3:
#             print(f"Detection {i}:")
#             print(f"  Class: {int(classes[0][i])}")
#             print(f"  Score: {scores[0][i]}")
#             print(f"  Box: {boxes[0][i]}")