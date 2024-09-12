#!/usr/bin/env python

import argparse
from pathlib import Path
import sys
from typing import List

import cv2
from PIL import Image, ImageDraw
from transformers import pipeline

from birds.lib.logger import get_logger, update_app_verbosity_level


logger = get_logger("main", verbosity=0)


def extract_clip(input_video_file, output_video_file, start_time, end_time):
    input_video_file = Path(input_video_file).resolve()
    logger.info(f"Extracting a clip from '{input_video_file}' to '{output_video_file}' ({start_time}-{end_time}s)")
    if not input_video_file.exists():
        logger.error(f"File '{input_video_file.name}' does not exist at '{input_video_file.parent}'")
        return 1

    if start_time > end_time:
        logger.error(f"Clip start time ({start_time}s) can't be larger than its end time ({end_time}s)")
        return 1

    # 'cap' is short for "capture" and refers to the video capture object created by cv2.VideoCapture.
    # This object allows us to interact with the video file (e.g., reading frames, getting video properties).
    cap = cv2.VideoCapture(str(input_video_file))

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
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

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

def split_video_clip_to_frames(video_file, output_dir):
    video_file = Path(video_file).resolve()
    if not video_file.exists():
        logger.error(f"File '{video_file.name}' does not exist at '{video_file.parent}'")
        return 1

    logger.info(f"Extracting image frames from '{video_file}'")

    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    logger.info(f"Saving frames to '{output_dir}'")

    cap = cv2.VideoCapture(str(video_file))
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_file = output_dir / f"frame{count:05d}.png"
        cv2.imwrite(str(output_file), frame)
        count += 1

    cap.release()
    logger.info(f"Extracted {count} frames from '{video_file.name}'")

    return 0

def detect_objects(input_image_file: str, output_image_file: str, candidate_labels: List[str]):
    input_image_file = Path(input_image_file).resolve()
    if not input_image_file.exists():
        logger.error(f"File '{input_image_file.name}' does not exist at '{input_image_file.parent}'")
        return 1

    checkpoint = "google/owlv2-base-patch16-ensemble"
    logger.info(f"Detecting objects: '{candidate_labels}' using '{checkpoint}' model checkpoint...")
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

    image = Image.open(str(input_image_file)).convert("RGB")
    predictions = detector(
        image,
        candidate_labels=candidate_labels,
    )
    logger.info(f"Result: {predictions}")

    draw = ImageDraw.Draw(image)
    for prediction in predictions:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        xmin, ymin, xmax, ymax = box.values()
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

    output_image_file = Path(output_image_file).resolve()
    if not output_image_file.parent.exists():
        output_image_file.parent.mkdir(parents=True)
    image.save(output_image_file)

    return 0

def create_cli():
    parser = argparse.ArgumentParser(description="Video processing and object detection tool")
    parser.add_argument("-v", action="count", dest="verbosity", help="Increase output verbosity (use -v, -vv, or -vvv for more detailed output)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_clip_parser = subparsers.add_parser("extract-clip",
                                                description="Extract a specific time segment from a video file",
                                                help="Extract a specific time segment from a video file")
    extract_clip_parser.add_argument("-i", "--input", required=True, help="Path to the input video file")
    extract_clip_parser.add_argument("-o", "--output", required=True, help="Path to save the output video file")
    extract_clip_parser.add_argument("--start", type=int, default=0, help="Start time of the clip in seconds (default: %(default)d)")
    extract_clip_parser.add_argument("--end", type=int, required=True, help="End time of the clip in seconds")

    extract_frames_parser = subparsers.add_parser("extract-frames",
                                                  description="Extract individual frames from a video file as images",
                                                  help="Extract individual frames from a video file as images")
    extract_frames_parser.add_argument("-i", "--input", required=True, help="Path to the input video file")
    extract_frames_parser.add_argument("-o", "--output-dir", required=True, help="Directory to save extracted frames")

    object_detection_parser = subparsers.add_parser("detect-objects",
                                                    description="Detect specified objects in an image",
                                                    help="Detect specified objects in an image")
    object_detection_parser.add_argument("-i", "--input", required=True, help="Path to the input image file")
    object_detection_parser.add_argument("-o", "--output", required=True, help="Path to save the output image file with detections")
    object_detection_parser.add_argument("--candidates",
                                         default="bird,flower pot",
                                         help="Comma-separated list of objects to detect (e.g., 'bird,car,person') Default: %(default)s")

    return parser

def main():
    cli = create_cli()
    args = cli.parse_args()
    update_app_verbosity_level(args.verbosity or 0)

    logger.debug(f"CLI call args: {args}")
    logger.debug(f"Work dir: {Path.cwd()}")

    status = 0
    if args.command == "extract-clip":
        status = extract_clip(args.input, args.output, args.start, args.end)
    elif args.command == "extract-frames":
        status = split_video_clip_to_frames(video_file=args.input, output_dir=args.output_dir)
    elif args.command == "detect-objects":
        candidate_labels = [obj.strip() for obj in args.candidates.split(',')]
        status = detect_objects(args.input, args.output, candidate_labels)

    sys.exit(status)

if __name__ == "__main__":
    main()
