import os
import json
import argparse
from tqdm import tqdm
from groundeddino_vl.api import load_model, load_image, predict
from . import review
from .config import logger, resolve_gdino_paths


def _normalize_annotations(annotations, frame_files):
    if not isinstance(annotations, dict):
        annotations = {}

    for frame_file in frame_files:
        frame_entry = annotations.get(frame_file, {})
        if not isinstance(frame_entry, dict):
            frame_entry = {}
        frame_entry.setdefault("objects", [])
        frame_entry.setdefault("Status", "pending")
        annotations[frame_file] = frame_entry

    return annotations


def run_gdino(
    input_dir,
    prompt,
    videoname,
    box_threshold=0.5,
    text_threshold=0.25,
    config_path=None,
    checkpoint_path=None,
):
    annotation_file = os.path.join(input_dir, f"annotation_{videoname}.json")

    # Load existing annotations if available
    if os.path.exists(annotation_file):
        with open(annotation_file, "r") as f:
            annotations = json.load(f)
        logger.info(f"Loaded existing annotations from {annotation_file}")
    else:
        annotations = {}

    # Gather all frame files
    frame_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))]
    )
    if not frame_files:
        logger.info("No frames found. Skipping inference.")
        return annotation_file

    annotations = _normalize_annotations(annotations, frame_files)

    # Identify frames to process
    frames_to_process = [f for f in frame_files if not annotations[f]['objects']]
    if not frames_to_process:
        logger.info("All frames already annotated. Skipping inference.")
        return annotation_file

    # Load model
    config_path, checkpoint_path = resolve_gdino_paths(config_path, checkpoint_path)
    logger.info("Loading GroundingDINO model...")
    model = load_model(config_path, checkpoint_path)
    logger.info("Model loaded successfully.")

    logger.info(f"Running inference on {len(frames_to_process)} new frames...")

    for frame_file in tqdm(frames_to_process, desc="Processing frames", unit="frame"):
        frame_path = os.path.join(input_dir, frame_file)
        image_source, image = load_image(frame_path)

        res = predict(
            model=model,
            image=image,
            text_prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        boxes, logits, phrases = res.boxes, res.scores, res.labels

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
    parser.add_argument("--videoname", required=False, help="Video name for annotation file")
    parser.add_argument(
        "--review",
        choices=["pending", "review", "none"],
        default="none",
        help="Open review tool after inference (pending, review, or none).",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.5,
        help="Box confidence threshold",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="Text confidence threshold",
    )
    parser.add_argument(
        "--config",
        help="Path to GroundingDINO config (overrides env)",
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to GroundingDINO checkpoint (overrides env)",
    )
    args = parser.parse_args()

    annotation_file = run_gdino(
        args.input_dir,
        args.prompt,
        args.videoname,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )

    if args.review in ("pending", "review"):
        review.start_review(args.input_dir, annotation_file, mode=args.review)


if __name__ == "__main__":
    main()
