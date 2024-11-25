#!/usr/bin/env zsh

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

interactive_mode=false  # Show results after running the script
for arg in "$@"; do
    case "$arg" in
        --interactive)
            interactive_mode=true
            ;;
        *)
            echo "Unknown option: $arg"
            ;;
    esac
done

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

if [[ $interactive_mode = true && $(uname) = "Darwin" ]]; then
    open test-cli/object-detection/prediction.jpg
fi
