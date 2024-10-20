import cv2

import numpy as np
import streamlit as st

from birds.lib.image_detection import (
    annotate_birds_and_other_animate_creatures,
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

    left_column, right_column = st.columns([2, 3])

    uploaded_image = None

    with left_column:
        uploaded_image = st.file_uploader(
            "Choose an image...",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded image", use_column_width=True)

    model = load_model()
    if uploaded_image is not None:
        uploaded_image_data = np.asarray(bytearray(uploaded_image.getvalue()))

        original_image_as_ndarray = cv2.imdecode(uploaded_image_data, cv2.IMREAD_COLOR)
        prepocessed = preprocess_image(original_image_as_ndarray, target_size=IMG_SIZE_FOR_DETECTOR)
        normalized = normalize_for_tf(image=prepocessed)
        detector_output = model(normalized)
        boxes = detector_output["detection_boxes"][0].numpy()
        classes = detector_output["detection_classes"][0].numpy()
        scores = detector_output["detection_scores"][0].numpy()

        annotated = annotate_birds_and_other_animate_creatures(
            image=original_image_as_ndarray,
            boxes=boxes,
            classes=classes,
            scores=scores,
        )
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with right_column:
            st.image(annotated, caption="To hell and back")


if __name__ == "__main__":
    start_app()
