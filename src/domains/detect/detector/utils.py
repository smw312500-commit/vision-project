import os
from pathlib import Path

import cv2

MODELS_FOLDER = Path(os.environ.get("MODELS_FOLDER", "model"))
DETECTOR_DEBUG = os.environ.get("DETECTOR_DEBUG") == "1"

# YOLO base models that ultralytics can auto-download
_YOLO_HUB_MODELS = {"yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt"}


def debug_print(*parts: object) -> None:
    if DETECTOR_DEBUG:
        print(*parts, flush=True)


def resolve_model_path(*model_names: str, purpose: str | None = None) -> Path:
    for model_name in model_names:
        model_path = MODELS_FOLDER / model_name
        if model_path.exists():
            return model_path

    # Fall back to ultralytics auto-download for known base models
    for model_name in model_names:
        if model_name in _YOLO_HUB_MODELS:
            return Path(model_name)

    purpose_suffix = f" for {purpose}" if purpose else ""
    expected = ", ".join(model_names)
    raise FileNotFoundError(
        f"Missing detector model{purpose_suffix}. Expected one of [{expected}] in {MODELS_FOLDER}."
    )


def open_video_capture(src: str | Path) -> tuple[cv2.VideoCapture, float, int, int]:
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"Input video not found: {src_path}")

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {src_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid video metadata for {src_path}: fps={fps}, size={width}x{height}"
        )

    return cap, fps, width, height


def create_video_writer(
    dest: str | Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(dest_path),
            cv2.VideoWriter.fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(
        f"Failed to open output writer for {dest_path} using codecs avc1/mp4v"
    )
