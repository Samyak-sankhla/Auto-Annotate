import os
import json
import cv2
import tkinter as tk
from tkinter import simpledialog
import argparse
from .config import logger

# ------------------- Helpers -------------------
def from_normalized_center(bbox, frame_w, frame_h):
    cx = int(bbox[0] * frame_w)
    cy = int(bbox[1] * frame_h)
    bw = int(bbox[2] * frame_w)
    bh = int(bbox[3] * frame_h)
    return cx, cy, bw, bh


def to_normalized_center(cx, cy, bw, bh, frame_w, frame_h):
    return [cx / frame_w, cy / frame_h, bw / frame_w, bh / frame_h]


def draw_annotations(frame, objects, status, preview_box=None, hovered_idx=None, mode="VIEW", zoom=1.0):
    orig_h, orig_w = frame.shape[:2]
    frame = cv2.resize(frame, (int(orig_w * zoom), int(orig_h * zoom)))
    h, w = frame.shape[:2]

    # Draw annotations
    for i, obj in enumerate(objects):
        cx, cy, bw, bh = from_normalized_center(obj["bbox"], orig_w, orig_h)
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2
        x1 = int(x1 * zoom)
        y1 = int(y1 * zoom)
        x2 = int(x2 * zoom)
        y2 = int(y2 * zoom)
        color = (0, 200, 0)
        if hovered_idx == i and mode == "DELETE":
            color = (0, 165, 255)  # highlight
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Only display label (no index)
        txt = obj['label']
        cv2.putText(frame, txt, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # Preview box
    if preview_box:
        x1, y1, x2, y2 = preview_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)

    # Status
    color = (0, 255, 0) if status == "approved" else (0, 0, 255)
    cv2.circle(frame, (25, 25), 6, color, -1)
    cv2.putText(frame, status.upper(), (45, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Mode: {mode}", (45, 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Footer
    footer_h = 40
    disp = cv2.copyMakeBorder(frame, 0, footer_h, 0, 0,
                              cv2.BORDER_CONSTANT, value=(30, 30, 30))
    footer_text = "[N] Next  [P] Prev  [A] Approve  [D] Delete  [C] Create  [Z] Zoom+  [X] Zoom-  [Q] Quit"
    text_size = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h + (footer_h // 2) + (text_size[1] // 2)
    cv2.putText(disp, footer_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return disp


# ------------------- Review Tool -------------------
def _normalize_annotations(annotations):
    if not isinstance(annotations, dict):
        return {}

    for frame_file, entry in list(annotations.items()):
        if not isinstance(entry, dict):
            entry = {}
        entry.setdefault("objects", [])
        entry.setdefault("Status", "pending")
        annotations[frame_file] = entry

    return annotations


def start_review(input_dir, annotation_file, mode="pending"):
    if not os.path.exists(annotation_file):
        logger.error(f"Annotation file not found: {annotation_file}")
        return

    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    annotations = _normalize_annotations(annotations)

    # Filter frames
    if mode == "pending":
        frame_files = [
            f
            for f, ann in annotations.items()
            if ann.get("Status", "pending") == "pending"
        ]
    else:  # "review"
        frame_files = list(annotations.keys())

    frame_files = sorted(
        [f for f in frame_files if os.path.exists(os.path.join(input_dir, f))]
    )

    if not frame_files:
        logger.info("No frames available for review.")
        return

    idx = 0
    box = {"start": None, "end": None}
    preview_box = None
    hovered_idx = None
    zoom = 1.0
    edit_mode = "VIEW"
    current_shape = None

    tk_root = tk.Tk()
    tk_root.withdraw()
    tk_root.attributes("-topmost", True)

    cv2.namedWindow("Review", cv2.WINDOW_NORMAL)

    def mouse_cb(event, x, y, flags, param):
        nonlocal preview_box, hovered_idx, edit_mode, box

        if current_shape is None:
            return

        orig_h, orig_w = current_shape
        ox = int(x / zoom)
        oy = int(y / zoom)

        if edit_mode == "CREATE":
            if event == cv2.EVENT_LBUTTONDOWN:
                box["start"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and box["start"]:
                box["end"] = (x, y)
                preview_box = (box["start"][0], box["start"][1], x, y)
            elif event == cv2.EVENT_LBUTTONUP and box["start"]:
                x1, y1 = box["start"]
                x2, y2 = x, y
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bw, bh = abs(x2 - x1), abs(y2 - y1)
                cx = int(cx / zoom)
                cy = int(cy / zoom)
                bw = int(bw / zoom)
                bh = int(bh / zoom)
                norm_box = to_normalized_center(cx, cy, bw, bh, orig_w, orig_h)
                label = simpledialog.askstring("Label", "Enter label:")
                if label:
                    new_obj = {"bbox": norm_box, "label": label}
                    annotations[frame_files[idx]]["objects"].append(new_obj)
                    with open(annotation_file, "w") as f:
                        json.dump(annotations, f, indent=2)
                    logger.info(f"Added {new_obj} to {frame_files[idx]}")
                box = {"start": None, "end": None}
                preview_box = None
                edit_mode = "VIEW"

        elif edit_mode == "DELETE":
            if event == cv2.EVENT_MOUSEMOVE:
                hovered_idx = None
                for i, obj in enumerate(annotations[frame_files[idx]]["objects"]):
                    cx, cy, bw, bh = from_normalized_center(obj["bbox"], orig_w, orig_h)
                    if cx - bw // 2 <= ox <= cx + bw // 2 and cy - bh // 2 <= oy <= cy + bh // 2:
                        hovered_idx = i
                        break
            elif event == cv2.EVENT_LBUTTONDOWN and hovered_idx is not None:
                obj = annotations[frame_files[idx]]["objects"].pop(hovered_idx)
                with open(annotation_file, "w") as f:
                    json.dump(annotations, f, indent=2)
                logger.info(f"Deleted {obj} from {frame_files[idx]}")
                hovered_idx = None
                edit_mode = "VIEW"

    cv2.setMouseCallback("Review", mouse_cb)

    while True:
        frame_file = frame_files[idx]
        frame = cv2.imread(os.path.join(input_dir, frame_file))
        if frame is None:
            frame_files.pop(idx)
            if not frame_files:
                logger.info("No readable frames available for review.")
                break
            idx = idx % len(frame_files)
            continue

        current_shape = frame.shape[:2]

        status = annotations[frame_file].get("Status", "pending")
        disp = draw_annotations(frame, annotations[frame_file]["objects"], status,
                                preview_box, hovered_idx, edit_mode, zoom)
        cv2.imshow("Review", disp)
        key = cv2.waitKey(50) & 0xFF

        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord("n"), ord("N")):
            idx = (idx + 1) % len(frame_files)
            edit_mode = "VIEW"
        elif key in (ord("p"), ord("P")):
            idx = (idx - 1) % len(frame_files)
            edit_mode = "VIEW"
        elif key in (ord("a"), ord("A")):  # approve
            annotations[frame_file]["Status"] = "approved"
            with open(annotation_file, "w") as f:
                json.dump(annotations, f, indent=2)
            logger.info(f"Frame {frame_file} approved.")

            if mode == "pending":
                frame_files.pop(idx)
                if not frame_files:
                    logger.info("All pending frames approved.")
                    break
                idx = idx % len(frame_files)
            else:
                idx = (idx + 1) % len(frame_files)
            edit_mode = "VIEW"
        elif key in (ord("d"), ord("D")):
            edit_mode = "DELETE"
        elif key in (ord("c"), ord("C")):
            edit_mode = "CREATE"
        elif key in (ord("z"), ord("Z")):
            zoom = min(zoom * 1.2, 5.0)
        elif key in (ord("x"), ord("X")):
            zoom = max(zoom / 1.2, 0.2)

    cv2.destroyAllWindows()
    tk_root.destroy()
    with open(annotation_file, "w") as f:
        json.dump(annotations, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Review annotations (interactive)")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--mode", choices=["pending", "review"], default="pending",
                        help="Show only pending frames (default) or all frames (review)")
    args = parser.parse_args()
    start_review(args.input_dir, args.annotation_file, mode=args.mode)


if __name__ == "__main__":
    main()
