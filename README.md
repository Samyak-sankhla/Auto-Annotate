
# Auto Annotation Pipeline

Auto Annotation Pipeline is a modular framework for automatically annotating video data. It combines frame extraction, duplicate removal, automatic annotation with GroundingDINO, and an interactive review GUI.

## Project Structure

```
autoannotation/
  auto_annotator/
    __init__.py
    config.py
    video_extractor.py
    dedup.py
    runner.py
    review.py
    pipeline.py
  pyproject.toml
  README.md
```

## Installation

```bash
git clone --recurse-submodules https://github.com/Samyak-sankhla/Auto-Annotate.git
cd autoannotation

python -m venv venv
venv\Scripts\activate

pip install -e GroundingDINO
pip install -e .
```

If you already cloned the repo without submodules:

```bash
git submodule update --init --recursive
```

### GroundingDINO setup

Download the GroundingDINO checkpoint and set the required paths as environment variables:

```powershell
$env:GROUNDING_DINO_CONFIG_PATH = "C:\path\to\Auto-Annotate\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
$env:GROUNDING_DINO_CHECKPOINT_PATH = "C:\path\to\groundingdino_swint_ogc.pth"
```

## Usage

All commands below are installed as CLI entry points after `pip install -e .`.

### 1) Run the whole pipeline with one command

```bash
auto-pipeline --input_video "C:\data\video.mp4" --output_dir "C:\data\output" --fps 1 --prompt "person"
```

Options:
- `--review pending|review|none` (default: `pending`)
- `--threshold 0.9` for dedup (lower is stricter)
- `--box_threshold` and `--text_threshold` for inference
- `--config` and `--checkpoint` to override env paths

### 2) Run auto-annotation on a frame directory and open the review GUI

```bash
auto-annotate --input_dir "C:\data\frames" --prompt "person" --videoname "video" --review review
```

Use `--review pending` to show only pending frames.

### 3) Run the review window for already annotated frames

```bash
auto-review --input_dir "C:\data\frames" --annotation_file "C:\data\frames\annotation_video.json" --mode review
```

### 4) Run the review window for pending annotations only

```bash
auto-review --input_dir "C:\data\frames" --annotation_file "C:\data\frames\annotation_video.json" --mode pending
```

### Frame extraction only

```bash
auto-extract "C:\data\video.mp4" 1 "C:\data\output"
```

### Deduplication only

```bash
auto-dedup --input_dir "C:\data\output\video_images" --threshold 0.9
```

## Review GUI Controls

The review window supports quick review and editing:

- `N` next frame, `P` previous frame
- `A` approve frame
- `D` delete bounding box (hover to highlight)
- `C` create bounding box
- `Z` zoom in, `X` zoom out
- `Q` quit

## Outputs

- Frames are saved into `<output_dir>\<video_name>_images`.
- Annotation JSON is saved in the frames directory as `annotation_<video_name>.json`.
- Deduplication logs are stored under `<frames_dir>\duplicates\logs`.

## Requirements

- Python 3.8+
- OpenCV (cv2)
- PyTorch (torch)
- NumPy
- Pandas
- fastdup
- tqdm
- Tkinter (bundled with most Python installs)
- GroundingDINO (included as submodule)

