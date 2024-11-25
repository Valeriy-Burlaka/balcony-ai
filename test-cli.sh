#!/usr/bin/env zsh

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

function clean_outputs() {
    rm test-cli/output.mp4
    rm -rf test-cli/4-sec-video-to-frames
    rm -rf test-cli/object-detection
}

clean_outputs

./src/birds/main.py -vv extract-clip -i test-cli/test-cli.mp4 -o test-cli/output.mp4  --start 1 --end 2
./src/birds/main.py -vv extract-frames -i test-cli/output.mp4 -o test-cli/4-sec-video-to-frames
./src/birds/main.py -vv detect-objects \
    -i test-cli/test-cli.jpg \
    -o test-cli/object-detection/prediction.jpg

open test-cli/object-detection/prediction.jpg
