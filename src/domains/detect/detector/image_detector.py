from pathlib import Path

from ultralytics import YOLO

from src.domains.detect.detector.utils import resolve_model_path


class BaseImageDetector:
    def __init__(self, model_src: Path, **kwargs):
        self.model = YOLO(str(model_src), **kwargs)

    def detect(self, src, **kwargs):
        return self.model(src, **kwargs)


class ImageDetectorYOLO11n(BaseImageDetector):
    def __init__(self):
        super().__init__(
            resolve_model_path("yolo11n.pt", purpose="ImageDetectorYOLO11n"),
            verbose=False,
        )


class ImageDetectorFireDetectV1(BaseImageDetector):
    def __init__(self):
        super().__init__(
            resolve_model_path(
                "fire_detect_v251205_1.pt", purpose="ImageDetectorFireDetectV1"
            ),
            verbose=False,
        )
