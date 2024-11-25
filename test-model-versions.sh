#!/usr/bin/env zsh

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

model_versions=(d1 d2 d3 d4 d5 d6 d7)
for model_version in "${model_versions[@]}"; do
    echo "Testing model version $model_version"
    ./src/birds/main.py -vvv detect-objects \
        -i test-cli/test-cli.jpg \
        -o "test-cli/object-detection/prediction-${model_version}.jpg" \
        --model-version $model_version
done

