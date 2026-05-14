from enum import Enum

from src.domains.detect.detector.image_detector import (
    BaseImageDetector,
    ImageDetectorYOLO11n,
    ImageDetectorFireDetectV1,
)
from src.domains.detect.detector.video_detector import (
    VideoDetectorFireDetectV1,
    BaseVideoDetector,
    VideoDetectorYolo11n,
    VideoDetectorShoulderStop,
    VideoDetectorWrongWay,
)
from src.domains.detect.detector.highway_detector import VideoDetectorHighway


class ImageDetectorEnum(Enum):
    YOLO11n = "yolo11n"
    FireDetectV1 = "fire_detect_v1"


image_detector_models = {
    ImageDetectorEnum.YOLO11n: ImageDetectorYOLO11n,
    ImageDetectorEnum.FireDetectV1: ImageDetectorFireDetectV1,
}


class VideoDetectorEnum(Enum):
    YOLO11n = "yolo11n"
    FireDetectV1 = "fire_detect_v1"
    ShoulderStop = "shoulder_stop"
    WrongWay = "wrong_way"
    Highway = "highway"


video_detector_models = {
    VideoDetectorEnum.YOLO11n: VideoDetectorYolo11n,
    VideoDetectorEnum.FireDetectV1: VideoDetectorFireDetectV1,
    VideoDetectorEnum.ShoulderStop: VideoDetectorShoulderStop,
    VideoDetectorEnum.WrongWay: VideoDetectorWrongWay,
    VideoDetectorEnum.Highway: VideoDetectorHighway,
}

__all__ = [
    "BaseImageDetector",
    "BaseVideoDetector",
    "ImageDetectorEnum",
    "ImageDetectorFireDetectV1",
    "ImageDetectorYOLO11n",
    "VideoDetectorEnum",
    "VideoDetectorFireDetectV1",
    "VideoDetectorHighway",
    "VideoDetectorShoulderStop",
    "VideoDetectorWrongWay",
    "VideoDetectorYolo11n",
    "image_detector_models",
    "video_detector_models",
]
