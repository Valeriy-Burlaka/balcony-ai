import argparse
import os

import cv2
from PIL import Image, ImageDraw
from transformers import pipeline

def extract_clip(input_video_path, output_video_path, start_time, end_time):
    # 'cap' is short for "capture" and refers to the video capture object created by cv2.VideoCapture.
    # This object allows us to interact with the video file (e.g., reading frames, getting video properties).
    cap = cv2.VideoCapture(input_video_path)

    # Retrieve the number of frames per second (fps) from the video, useful for timing calculations.
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video FPS:", fps)

    print("Skip to second ", start_time)
    # Calculate the start position in the video in milliseconds based on the given start time in seconds.
    start_msec = start_time * 1000
    # Set the current position of the video capture to 'start_msec', which allows us to start reading from this point.
    cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)

    # Define the codec for the output video file. 'fourcc' refers to "four character code".
    # It is a code that helps specify the video codec used in the functions of VideoWriter or VideoCapture.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Create a VideoWriter object to write the video to a file. We specify the codec, fps, and resolution.
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

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


def video_to_frames(path_to_video, output_frames_dir):
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    cap = cv2.VideoCapture(path_to_video)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_frames_dir, f'frame{count:05d}.png'), frame)
        count += 1

    cap.release()

def detect_objects(image_path, output_path, candidate_labels=["bird", "flower pot"]):
    checkpoint = "google/owlv2-base-patch16-ensemble"
    print(f"Detecting objects using {checkpoint} model checkpoint...")
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

    image = Image.open(image_path).convert("RGB")
    predictions = detector(
        image,
        candidate_labels=candidate_labels,
    )
    print(predictions)

    draw = ImageDraw.Draw(image)
    for prediction in predictions:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        xmin, ymin, xmax, ymax = box.values()
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

    image.save(output_path)

def create_parser():
    parser = argparse.ArgumentParser(description="Extract a clip from a video file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_clip_parser = subparsers.add_parser("extract-clip", help="Extract a clip from a video file.")
    extract_clip_parser.add_argument("-i", "--input", required=True, help="Input video file path")
    extract_clip_parser.add_argument("-o", "--output", required=True, help="Output video file path")
    extract_clip_parser.add_argument("--start", type=int, default=0, help="Start time in seconds (default=0)")
    extract_clip_parser.add_argument("--end", type=int, required=True, help="End time in seconds")
    extract_clip_parser.set_defaults(func=extract_clip)

    extract_frames_parser = subparsers.add_parser("extract-frames", help="Extract frames from a video clip.")
    extract_frames_parser.add_argument("-i", "--input", required=True, help="Input video file path")
    extract_frames_parser.add_argument("-o", "--output-dir", required=True, help="Output frames directory")
    extract_frames_parser.set_defaults(func=video_to_frames)

    object_detection_parser = subparsers.add_parser("detect-objects", help="Detect if candidate objects are present in the image")
    object_detection_parser.add_argument("-i", "--input", required=True, help="Input image file path")
    object_detection_parser.add_argument("-o", "--output", required=True, help="Output image file path with detections")
    object_detection_parser.add_argument("--candidates", required=False, help="Comma-separated list of candidate objects to detect")
    object_detection_parser.set_defaults(func=detect_objects)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    print("Args:", args)
    print(f"Work dir: {os.getcwd()}")

    if args.command == "extract-clip":
        args.func(args.input, args.output, args.start, args.end)
    elif args.command == "extract-frames":
        args.func(args.input, args.output_dir)
    elif args.command == "detect-objects":
        candidate_labels = [obj.strip() for obj in args.candidates.split(',')]
        args.func(args.input, args.output, candidate_labels)

if __name__ == "__main__":
    main()

