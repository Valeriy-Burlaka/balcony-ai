# Example usage:
#
# model = load_model("efficientdet", "d1")
from pathlib import Path

import kagglehub

import tensorflow as tf

from birds.lib.utils import timeit


def _download_model(family: str, version: str) -> str:
    with timeit(f"Downloading model {family}/{version}"):
        model_path = kagglehub.model_download(f"tensorflow/{family}/tensorFlow2/{version}")

    return model_path

def _get_model_path(family: str, version: str) -> str:
    local_path = Path(f"~/.cache/kagglehub/models/tensorflow/{family}/tensorFlow2/{version}/1").expanduser()
    if not local_path.exists():
        local_path = _download_model(family, version)

    return local_path

def load_model(family: str, version: str):
    model_path = _get_model_path(family, version)
    with timeit(f"Loading model {family}/{version}"):
        model = tf.saved_model.load(model_path)

    return model
