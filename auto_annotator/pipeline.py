import os
import argparse
from .config import logger
from .video_extractor import extract_frames
from .dedup import clean_image_duplicates
from .runner import run_gdino
from .review import start_review


def run_pipeline(
    input_video,
    output_dir,
    fps,
    prompt,
    threshold=0.9,
    review_mode="pending",
    videoname=None,
    box_threshold=0.5,
    text_threshold=0.25,
    config_path=None,
    checkpoint_path=None,
):
    if not os.path.isfile(input_video):
        logger.error(f"Input video not found: {input_video}")
        return None

    frames_dir = extract_frames(input_video, fps, output_dir)
    if not frames_dir:
        logger.error("Frame extraction failed.")
        return None

    clean_image_duplicates(input_dir=frames_dir, threshold=threshold)

    videoname = videoname or os.path.splitext(os.path.basename(input_video))[0]
    annotation_file = run_gdino(
        frames_dir,
        prompt,
        videoname,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    if review_mode in ("pending", "review"):
        start_review(frames_dir, annotation_file, mode=review_mode)

    return annotation_file


def main():
    parser = argparse.ArgumentParser(description="Run the full auto-annotation pipeline")
    parser.add_argument("--input_video", required=True, help="Path to input video")
    parser.add_argument("--output_dir", required=True, help="Output folder for frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction FPS")
    parser.add_argument("--prompt", required=True, help="Text prompt for detection")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Deduplication threshold (lower is stricter)",
    )
    parser.add_argument(
        "--review",
        choices=["pending", "review", "none"],
        default="pending",
        help="Review mode after inference",
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
    parser.add_argument(
        "--videoname",
        help="Annotation file prefix (defaults to video name)",
    )
    args = parser.parse_args()

    run_pipeline(
        input_video=args.input_video,
        output_dir=args.output_dir,
        fps=args.fps,
        prompt=args.prompt,
        threshold=args.threshold,
        review_mode=args.review,
        videoname=args.videoname,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
