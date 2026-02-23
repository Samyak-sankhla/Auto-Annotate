import cv2
import os
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def is_video_file(filename):
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS

def extract_frames(video_path, target_fps, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, f"{video_name}_images")
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        logging.error(f"Could not determine FPS for {video_path}")
        return

    if target_fps <= 0:
        logging.error("Target FPS must be > 0")
        return None

    ratio = original_fps / target_fps
    frame_interval = max(int(round(ratio)), 1)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"Processing video: {video_name}")
    logging.info(f"Total frames: {frame_count}")
    logging.info(f"Original FPS: {original_fps:.2f}, Target FPS: {target_fps}")
    logging.info(f"Frame interval: {frame_interval}")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = os.path.join(save_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    logging.info(f"Saved {saved_count} frames to {save_dir}")
    return save_dir

def process_input(input_path, target_fps, output_dir=None):
    if not os.path.exists(input_path):
        logging.error(f"Input path does not exist: {input_path}")
        return []

    # Default: always make a new "output" folder in current working directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        if not is_video_file(input_path):
            logging.error(f"Unsupported file type: {input_path}")
            return []
        save_dir = extract_frames(input_path, target_fps, output_dir)
        return [save_dir] if save_dir else []

    elif os.path.isdir(input_path):
        save_dirs = []
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path) and is_video_file(filename):
                save_dir = extract_frames(file_path, target_fps, output_dir)
                if save_dir:
                    save_dirs.append(save_dir)
            else:
                logging.warning(f"Skipping unsupported file: {filename}")
        return save_dirs
    else:
        logging.error(f"Invalid input: {input_path}")
        return []

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video(s) at given FPS"
    )
    parser.add_argument("input_path", help="Path to a video file or a directory containing videos")
    parser.add_argument("fps", type=float, help="FPS at which to save images")
    parser.add_argument("output_dir", nargs="?", help="Optional output directory (default: ./output)")

    args = parser.parse_args()
    process_input(args.input_path, args.fps, args.output_dir)

if __name__ == "__main__":
    main()
         