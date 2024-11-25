#!/usr/bin/env zsh

ENV_DIR=.tf-venv

if [[ ! -d $ENV_DIR ]]; then
    python -m venv $ENV_DIR
fi

source $ENV_DIR/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

# Verify installation
python3 -c "import tensorflow as tf; print('Random tensor test result: ', tf.reduce_sum(tf.random.normal([1000, 1000])))"

python3 -c "import tensorflow as tf; print('GPU: ', tf.config.list_physical_devices('GPU'))"

python3 -c "import tensorflow as tf; print('CPU: ', tf.config.list_physical_devices('CPU'))"
