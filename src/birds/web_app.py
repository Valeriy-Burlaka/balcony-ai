import cv2

import numpy as np
import streamlit as st

from PIL import Image
from streamlit_cropper import st_cropper

from birds.lib.image_detection import (
    annotate_image_with_selected_classes,
    IMG_SIZE_FOR_DETECTOR,
)
from birds.lib.image_preprocessing import normalize_for_tf, preprocess_image
from birds.lib.logger import get_logger, update_app_verbosity_level
from birds.lib.tf_models import load_model as _load_model


logger = get_logger("web_app", verbosity=2)
update_app_verbosity_level(verbosity=2)


@st.cache_resource
def load_model():
    return _load_model(family="efficientdet", version="d6")


def start_app():
    st.set_page_config(
        layout="wide",
        page_title="Balcony life",
        page_icon="🦉"
    )
    st.header("Turn any Python script into a compelling web app 🐦🐦‍⬛🐓🦉🦅")
    st.button("Rerun")

    left_column, right_column = st.columns(2)

    st.write("Double click to save crop")
    uploaded_image = st.sidebar.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg"],
    )
    cropped_image = None

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        with left_column:
            cropped_image = st_cropper(
                img_file=img,
                aspect_ratio=(1, 1),
                realtime_update=False,
                # should_resize_image=False,
                stroke_width=2,
            )
            # st.image(uploaded_image, caption="Uploaded image", use_column_width=True)
        with left_column:
            _ = cropped_image.thumbnail((IMG_SIZE_FOR_DETECTOR, IMG_SIZE_FOR_DETECTOR))
            st.image(cropped_image, caption="Cropped image")

    model = load_model()

    print("Cropped image: ", cropped_image)
    if uploaded_image is not None:
        uploaded_image_data = np.asarray(bytearray(uploaded_image.getvalue()))

        original_image_as_ndarray = cv2.imdecode(uploaded_image_data, cv2.IMREAD_COLOR)
        prepocessed = preprocess_image(original_image_as_ndarray, target_size=IMG_SIZE_FOR_DETECTOR)
        normalized = normalize_for_tf(image=prepocessed)
        detector_output = model(normalized)
        boxes = detector_output["detection_boxes"][0].numpy()
        classes = detector_output["detection_classes"][0].numpy()
        scores = detector_output["detection_scores"][0].numpy()

        annotated = annotate_image_with_selected_classes(
            image=original_image_as_ndarray,
            boxes=boxes,
            classes=classes,
            scores=scores,
        )
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with right_column:
            st.image(annotated, caption="To hell and back")

    if cropped_image is not None:
        cropped_image_as_ndarray = np.array(cropped_image, dtype=np.uint8)
        print(cropped_image_as_ndarray.shape)

        # prepocessed = preprocess_image(cropped_image_as_ndarray, target_size=IMG_SIZE_FOR_DETECTOR)
        normalized = normalize_for_tf(image=cropped_image_as_ndarray)
        detector_output = model(normalized)
        boxes = detector_output["detection_boxes"][0].numpy()
        classes = detector_output["detection_classes"][0].numpy()
        scores = detector_output["detection_scores"][0].numpy()

        print("Classes", classes)
        print("Scores", scores)
        print("Boxes", boxes)


        annotated = annotate_image_with_selected_classes(
            image=cropped_image_as_ndarray,
            boxes=boxes,
            classes=classes,
            scores=scores,
        )
        # annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with right_column:
            st.image(annotated, caption="To hell and back (twice)")


if __name__ == "__main__":
    start_app()
