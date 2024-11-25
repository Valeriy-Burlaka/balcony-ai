#!/usr/bin/env python

import argparse
import sys
import time

from pathlib import Path

import cv2

# from PIL import Image, ImageDraw

from birds.lib.logger import get_logger, update_app_verbosity_level, update_logger_verbosity_level
from birds.lib.tf_models import load_model
from birds.lib.image_detection import detect


logger = get_logger("main", verbosity=0)


def extract_clip(input_video_file, output_video_file, start_time, end_time):
    input_video_file = Path(input_video_file).resolve()
    output_video_file = Path(output_video_file).resolve().with_suffix(".mp4")
    logger.info(f"Extracting a clip from '{input_video_file}' to '{output_video_file}' ({start_time}-{end_time}s)")

    if not input_video_file.exists() or not input_video_file.is_file():
        logger.error(f"File '{input_video_file.name}' does not exist at '{input_video_file.parent}'")
        return 1

    if start_time > end_time:
        logger.error(f"Clip start time ({start_time}s) can't be larger than its end time ({end_time}s)")
        return 1

    # 'cap' is short for "capture" and refers to the video capture object created by cv2.VideoCapture.
    # This object allows us to interact with the video file (e.g., reading frames, getting video properties).
    cap = cv2.VideoCapture(input_video_file.as_posix())

    # Retrieve the number of frames per second (fps) from the video, useful for timing calculations.
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Input video FPS: {fps}")

    logger.debug(f"Skip to second: {start_time}")
    # Calculate the start position in the video in milliseconds based on the given start time in seconds.
    start_msec = start_time * 1000
    # Set the current position of the video capture to 'start_msec', which allows us to start reading from this point.
    cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)

    # Define the codec for the output video file. 'fourcc' refers to "four character code".
    # It is a code that helps specify the video codec used in the functions of VideoWriter or VideoCapture.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Create a VideoWriter object to write the video to a file. We specify the codec, fps, and resolution.
    out = cv2.VideoWriter(output_video_file.as_posix(), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Calculate the end position in milliseconds for the extraction.
    end_msec = end_time * 1000

    # Loop until we reach the end of the segment we want to extract.
    while True:
        # Stop reading if the current video position is beyond the end time.
        if cap.get(cv2.CAP_PROP_POS_MSEC) > end_msec:
            break
        # Read a frame from the video. 'ret' ('return') is a boolean that indicates
        #  if the frame was read successfully. 'frame' is the actual image data o
        # the frame read from the video.
        ret, frame = cap.read()
        # If the frame is not read successfully, stop the loop.
        if not ret:
            break
        # Write the successfully read frame to the output video.
        out.write(frame)

    # Release the video capture and the video writer objects to free up system resources.
    cap.release()
    out.release()

    return 0

def split_video_clip_to_frames(video_file, output_dir, image_format):
    if image_format.lower() not in ["jpg", "png"]:
        logger.error(f"Not supported image format: '{image_format}'")
        return 1

    video_file = Path(video_file).resolve()
    if not video_file.exists():
        logger.error(f"File '{video_file.name}' does not exist at '{video_file.parent}'")
        return 1

    logger.info(f"Extracting image frames from '{video_file}'")

    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    logger.info(f"Saving frames to '{output_dir}'")

    cap = cv2.VideoCapture(video_file.as_posix())
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_file = output_dir / f"{video_file.with_suffix('').name}__frame{count:05d}.{image_format}"
        cv2.imwrite(str(output_file), frame)
        count += 1

    cap.release()
    logger.info(f"Extracted {count} frames from '{video_file.name}'")

    return 0

def extract_clip_and_frames(input_video_file, start_time, end_time, output_dir, output_image_format):
    logger.info(f"Extracting video clip and frames from '{start_time}' s. to '{end_time}' s.")

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    input_path = Path(input_video_file)
    output_filename = f"{input_path.stem}__clip_{end_time - start_time}-sec{input_path.suffix}"
    output_video_file = output_dir / output_filename
    status = extract_clip(
        input_video_file=input_video_file,
        output_video_file=output_video_file,
        start_time=start_time,
        end_time=end_time,
    )
    if status == 0:
        status = split_video_clip_to_frames(
            video_file=output_video_file,
            output_dir=output_dir,
            image_format=output_image_format,
        )

    # Rename the clip so it appears at the top of the output dir, and my laptop
    # doesn't burn from scrolling down 1000s of item in file explorer. The line
    # below simply adds an underscore "_" to the file name.
    output_video_file.rename(output_video_file.with_stem(f"_{output_video_file.stem}"))

    return status

def detect_objects(input_image: str, output_image: str, model_family="efficientdet", model_version="d1"):
    image_file = Path(input_image).resolve()
    if not image_file.exists():
        logger.error(f"File '{image_file.name}' does not exist at '{image_file.parent}'")
        return 1

    logger.info(f"Processing image '{image_file}' with '{model_family}/{model_version}'")

    model = load_model(family=model_family, version=model_version)
    result = detect(model, image_file)

    output_path = Path(output_image).resolve()
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    cv2.imwrite(f"{output_path.as_posix()}", result.annotated_image)

    return 0

class InvalidTimeFormatError(argparse.ArgumentTypeError):
    def __init__(self, time_string: str):
        self.message = (
            f"Invalid time format: '{time_string}'. Please use 'MM:SS' "
            "or 'HH:MM:SS' format (HH:[00-23], MM:[00-59], SS: [00:59])")
        super.__init__(self.message)

    def __str__(self):
        return self.message

def timestring_to_total_seconds(time_string: str) -> int:
    num_time_parts = len(time_string.split(":"))
    if num_time_parts < 2:
        raise InvalidTimeFormatError(time_string)

    format_string = "%M:%S" if num_time_parts == 2 else "%H:%M:%S"
    try:
        t = time.strptime(time_string, format_string)

        return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec
    except ValueError:
        raise InvalidTimeFormatError(time_string)

def attach_extract_clip_parser(subparsers: argparse._SubParsersAction):
    # TODO: Add explanation why I'm doing both 'help' and 'description' (vaguely remember there was some issue with displaying help messages)
    parser = subparsers.add_parser("extract-clip",
                                    description="Extract a specific time segment from a video file",
                                    help="Extract a specific time segment from a video file")

    parser.add_argument("-i", "--input", required=True, help="Path to the input video file")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output video file")
    parser.add_argument("--start", type=int, default=0, help="Start time of the clip in seconds (default: %(default)d)")
    parser.add_argument("--end", type=int, required=True, help="End time of the clip in seconds")

    return parser

def attach_extract_frames_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("extract-frames",
                                    description="Extract individual frames from a video file as images",
                                    help="Extract individual frames from a video file as images")

    parser.add_argument("-i", "--input", required=True, help="Path to the input video file")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save extracted frames")
    parser.add_argument("-f", "--format", choices=["jpg", "png"], default="jpg", help="Output image file format")

def attach_extract_all_parser(subparsers: argparse._SubParsersAction):
    help_string = "Extract a specific time segment from a video file as a clip and images"
    parser = subparsers.add_parser("extract-all", description=help_string, help=help_string)

    parser.add_argument("-i", "--input", required=True, help="Path to the input video file")
    parser.add_argument("-o", "--output", required=True,
                        help=("Output dir to save the resulting video clip and image "
                              "frames. Will be created if not exists"))
    parser.add_argument("--start", required=True,
                        type=timestring_to_total_seconds,
                        help="Start time of the clip (format: MM:SS or HH:MM:SS)")
    parser.add_argument("--end", required=True,
                        type=timestring_to_total_seconds,
                        help="End time of the clip (format: MM:SS or HH:MM:SS)")
    parser.add_argument("-f", "--format", choices=["jpg", "png"], default="jpg",
                        help="Output image file format")


def create_cli():
    parser = argparse.ArgumentParser(description="Video processing and object detection tool")
    parser.add_argument("-v", action="count", dest="verbosity", help="Increase output verbosity (use -v, -vv, or -vvv for more detailed output)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    attach_extract_clip_parser(subparsers)

    attach_extract_frames_parser(subparsers)

    attach_extract_all_parser(subparsers)

    object_detection_parser = subparsers.add_parser("detect-objects",
                                                    description="Detect specified objects in an image using pipeline",
                                                    help="Detect specified objects in an image using pipeline")
    object_detection_parser.add_argument("-i", "--input", required=True, help="Path to the input image file")
    object_detection_parser.add_argument("-o", "--output", required=True, help="Path to save the output image file with detections")
    # model version: d1, d2, d3, d4, d5, d6, d7 # d4 is buggy — zero detection rate. Add arg: --model-version
    object_detection_parser.add_argument("--model-version",
                                         choices=["d1", "d2", "d3", "d4", "d5", "d6", "d7"],
                                         default="d1",
                                         help="Model version to use for object detection")

    return parser

def main() -> int:
    cli = create_cli()
    args = cli.parse_args()

    app_verbosity = args.verbosity or 0

    update_app_verbosity_level(app_verbosity)
    update_logger_verbosity_level(logger, app_verbosity)

    logger.debug(f"CLI call args: {args}")
    logger.debug(f"Work dir: {Path.cwd()}")

    status = 0

    if args.command == "extract-clip":
        status = extract_clip(args.input, args.output, args.start, args.end)
    elif args.command == "extract-frames":
        status = split_video_clip_to_frames(
            video_file=args.input,
            output_dir=args.output_dir,
            image_format=args.format)
    elif args.command == "extract-all":
        if args.start > args.end:
            logger.error(f"Start time ({args.start}) must be less than end time ({args.end})")
            return 1

        status = extract_clip_and_frames(
            input_video_file=args.input,
            start_time=args.start,
            end_time=args.end,
            output_dir=args.output,
            output_image_format=args.format,
        )
    elif args.command == "detect-objects":
        status = detect_objects(
            input_image=args.input,
            output_image=args.output,
            model_version=args.model_version,
        )

    return status

if __name__ == "__main__":
    status_code = main()
    sys.exit(status_code)
