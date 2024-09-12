#!/usr/bin/env sh

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

./src/birds/main.py -vv extract-clip -i test-detection/video-clips/test-cli.mp4 -o output.mp4  --start 1 --end 5
./src/birds/main.py -vv extract-frames -o test-cli/4-sec-video-to-frames -i output.mp4
./src/birds/main.py -vv detect-objects -i test-cli/4-sec-video-to-frames/frame00005.png -o test-cli/object-detection/frame00005_processed_flower-pot.png --candidates "garden pot,flower pot"
open test-cli/object-detection/frame00005_processed_flower-pot.png
