import os
import json
import cv2
import torch
import argparse
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict
from . import review
from .config import logger, GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH


def run_gdino(input_dir, prompt, videoname):
    annotation_file = os.path.join(input_dir, f"annotation_{videoname}.json")

    # Load existing annotations if available
    if os.path.exists(annotation_file):
        with open(annotation_file, "r") as f:
            annotations = json.load(f)
        logger.info(f"Loaded existing annotations from {annotation_file}")
    else:
        annotations = {}

    # Gather all frame files
    frame_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])

    # Identify frames to process
    frames_to_process = [f for f in frame_files if f not in annotations]
    if not frames_to_process:
        logger.info("All frames already annotated. Skipping inference.")
        return annotation_file

    # Load model
    logger.info("Loading GroundingDINO model...")
    model = load_model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)
    logger.info("Model loaded successfully.")

    logger.info(f"Running inference on {len(frames_to_process)} new frames...")

    for frame_file in tqdm(frames_to_process, desc="Processing frames", unit="frame"):
        frame_path = os.path.join(input_dir, frame_file)
        image_source, image = load_image(frame_path)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=0.5,
            text_threshold=0.25
        )

        annotations[frame_file] = {
            "objects": [],
            "Status": "pending"
        }
        for box, _, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
            annotations[frame_file]["objects"].append({
                "bbox": box,
                "label": phrase
            })

    # Save/update annotations
    with open(annotation_file, "w") as f:
        json.dump(annotations, f, indent=4)

    logger.info(f"Annotations saved/updated in {annotation_file}")
    return annotation_file


def main():
    parser = argparse.ArgumentParser(description="Run GroundingDINO on frames")
    parser.add_argument("--input_dir", required=True, help="Directory with frames")
    parser.add_argument("--prompt", required=True, help="Text prompt for detection")
    parser.add_argument("--videoname", required=True, help="Video name for annotation file")
    parser.add_argument("--review", nargs="?", const="pending",
                        help="Run review tool (optional). Default shows only pending, use 'review' to show all.")
    args = parser.parse_args()

    annotation_file = run_gdino(args.input_dir, args.prompt, args.videoname)

    if args.review is not None:
        review.start_review(args.input_dir, annotation_file, mode=args.review)


if __name__ == "__main__":
    main()
