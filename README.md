
# Auto Annotation Pipeline

The **Auto Annotation Pipeline** is a modular framework for automatically annotating video data using computer vision models.  
It combines **frame extraction**, **duplicate removal**, **automatic annotation with GroundingDINO**, and an **interactive review tool**.

---

## 🚀 Pipeline Overview

The pipeline follows these stages:

1. **Frame Extraction (`video_extractor.py`)**
   - Extracts frames from input videos at a target FPS.  
   - Saves frames into a structured directory (`<video_name>_images`).  
   - Supports multiple video formats (`.mp4`, `.avi`, `.mov`, `.mkv`).

2. **Deduplication (`dedup.py`)**
   - Uses [`fastdup`](https://github.com/visualdatabase/fastdup) to find and remove duplicate frames.  
   - Clusters similar images and moves duplicates into a separate folder.  
   - Generates logs and summaries for traceability.

3. **Auto Annotation (`runner.py`)**
   - Runs the [`GroundingDINO`](https://github.com/IDEA-Research/GroundingDINO) object detection model on extracted frames.  
   - Accepts a **text prompt** for class-specific detection.  
   - Produces a JSON file with bounding boxes, labels, and confidence scores.  
   - Automatically launches the review stage after inference.

4. **Review Tool (`review.py`)**
   - Provides an interactive GUI (OpenCV + Tkinter) to:  
     - Accept/reject detections.  
     - Manually create bounding boxes.  
     - Delete incorrect detections.  
   - Saves **final annotations** 

---

## 📂 Project Structure

```
auto_annotator/                # repository root
├─ auto_annotator/             # Python package
│  ├─ __init__.py
│  ├─ config.py
│  ├─ video_extractor.py       # Extract frames from video(s)
│  ├─ dedup.py                 # Deduplicate frames with fastdup
│  ├─ runner.py                # Run GroundingDINO inference + launch review
│  ├─ review.py                # Interactive annotation review tool
├─ pyproject.toml              # Build system + dependencies

```

---

## ⚡ Installation

```bash
# Clone repository
git clone <your-repo-url>
cd auto_annotator

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

#virtual environment through conda (preferred)
conda create -n gdino python=3.10 -y
conda activate gdino

# Install dependencies
pip install -e . # (run this code where .toml file is present. This will only install our module, not the grounding_dino module.)

---

## 📜 Usage

### 1. Extract frames from a video
```bash
auto-extract "Input_video" "fps" "output_directory"
```
Extracts frames at **1 FPS**.

### 2. Deduplicate frames
```bash
auto-dedup --input_dir "input-directory" [--threshold <float value between 0-1>]
```

### 3. Run GroundingDINO auto-annotation
```bash
auto-annotate --input_dir "input-directory" --prompt "prompt" --videoname "name" --review "review mode"
```

## ⚙️ Requirements

- Python **>=3.8**
- OpenCV (cv2)
- PyTorch (torch)
- NumPy
- Pandas
- Fastdup
- tqdm
- Tkinter (comes pre-installed with Python on most systems)
- GroundingDINO (installed separately from source)

---

## 🛠 Development Notes

- GroundingDINO requires a config + checkpoint file (update paths in `runner.py`).  
- The deduplication process may be GPU-heavy depending on dataset size.  

---

