#!/usr/bin/env sh

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"


function tear_down() {
    rm test-cli/output.mp4
    rm -rf test-cli/4-sec-video-to-frames
    rm -rf test-cli/object-detection
}

tear_down

./src/birds/main.py -vv extract-clip -i test-cli/test-cli.mp4 -o test-cli/output.mp4  --start 1 --end 2
./src/birds/main.py -vv extract-frames -i test-cli/output.mp4 -o test-cli/4-sec-video-to-frames
./src/birds/main.py -vv detect-objects \
    -i test-cli/4-sec-video-to-frames/output__frame00015.jpg \
    -o test-cli/object-detection/frame00015_processed_flower-pot.jpg \
    --candidates "garden pot,flower pot"
open test-cli/object-detection/frame00015_processed_flower-pot.jpg
