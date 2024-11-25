#!/usr/bin/env zsh

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

python src/birds/test_tf_model.py ./test-detection/datasets/functional-regression-test
