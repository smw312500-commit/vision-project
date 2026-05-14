"""고속도로 이상 주행 감지 Detector (역주행 / 정차 / 정체)
빨간 박스 표시 + JSON 결과 반환

BaseVideoDetector 인터페이스와 다르게 detect()가 dict를 반환합니다.
"""

import time
import traceback
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.domains.detect.detector.utils import (
    create_video_writer,
    debug_print,
    resolve_model_path,
)

try:
    from sklearn.cluster import KMeans
except ImportError:
    raise ImportError("scikit-learn 필요: uv add scikit-learn")

# ──────────────────────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────────────────────
CONF_THRESH      = 0.35
VEHICLE_CLASSES  = [2, 3, 5, 7]   # car, motorcycle, bus, truck
TRACKER_CFG      = 'bytetrack.yaml'   # YOLO 내부 tracker 사용
TRACK_PERSIST     = True

MAX_DIST         = 80
MAX_LOST         = 10
MIN_HITS         = 8
TRAIL_LEN        = 50
MIN_TRAJ_LEN     = 15
CLUSTER_EVERY    = 15
MIN_VALID_TRACKS = 8

STOP_SPEED_PX    = 1.5
# 정차: 혼자 10초 이상 안 움직임
STOP_S           = 10.0            # 단독 정차 확정 시간(초)

# 정체: 2초 이상 멈춘 차가 반경 내 N대 이상
CONGESTION_S       = 2.0           # 정체 판단용 멈춤 시간(초)
CONGESTION_RADIUS  = 120           # px
CONGESTION_MIN_N   = 3             # 반경 내 멈춘 차 최소 대수

# 역주행 확정/해제 히스테리시스 (초 기준)
WRONG_CANDIDATE_S = 0.5   # 0.5초 반대 움직임이면 강한 후보
WRONG_CONFIRM_S   = 1.2   # 1.2초 지속 시 확정
WRONG_RELEASE_S   = 0.5   # 확정 후 0.5초 이하 수준으로 떨어지면 해제
WRONG_DECAY       = 1
WRONG_DOT_THRESH  = 0.4   # road_axis dot 부호 반전 임계값

# 새 wrong-way 후보 정의용 보강 파라미터
WRONG_CENTER_MARGIN_DEG = 0.0   # 반대 중심이 자기 중심보다 최소 이 각도만큼 더 가까워야 함
WRONG_MIN_SPEED         = 1.2    # 너무 느리면 제외
WRONG_MIN_DISP_RATIO    = 0.15   # recent vector 상대변위 최소값 (disp / bbox_height)
WRONG_VEC_WIN           = 16     # 최근 벡터 계산 window (먼 차량용으로 증가)

# road_axis 부호 초기화
AXIS_INIT_S      = 0.0

SWAP_GUARD_DEG   = 60.0
FLOW_EMA_ALPHA   = 0.05

# dir EMA 히스테리시스
DIR_SCORE_UP     =  0.35
DIR_SCORE_DOWN   = -0.35
DIR_EMA_ALPHA    =  0.10

# 색상 (BGR)
COLOR_ANOMALY    = (0,   0, 255)   # 빨강 - 역주행/정차
COLOR_CONGESTION = (0, 200, 255)   # 노랑 - 정체
COLOR_UP         = (50, 220,  50)  # 초록 - 정상 상행
COLOR_DOWN       = (50,  50, 255)  # 파랑 - 정상 하행
COLOR_UNKNOWN    = (180, 180, 180) # 회색
COLOR_ZONE_UP    = (255, 80, 80)   # 파란 계열 바닥 (BGR)
COLOR_ZONE_DOWN  = (80, 255, 80)   # 녹색 계열 바닥 (BGR)
ZONE_ALPHA       = 0.18

ANOMALY_LABEL = {
    'WRONG_WAY':   'WRONG WAY',
    'STOPPED':     'STOPPED',
    'CONGESTION':  'CONGESTION',
}
CLS_NAME = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


# ──────────────────────────────────────────────────────────────
# progress / failure helpers
# ──────────────────────────────────────────────────────────────
def _stage_print(stage: str, msg: str):
    debug_print(f"[highway_detector][{stage}] {msg}")

def _write_error_sidecar(dest: Path, payload: dict):
    try:
        err_path = Path(str(dest) + '.error.json')
        import json
        err_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def _precheck_before_detect(src: Path, dest: Path, model_obj):
    """
    탐지 시작 전 선행 체크.
    source / metadata / model / writer를 먼저 확인해서
    실패 원인을 모델 문제인지 Flask/IO 문제인지 분리한다.
    """
    checks = {
        "src_exists": False,
        "src_openable": False,
        "video_meta_ok": False,
        "dest_parent_exists": False,
        "writer_openable": False,
        "model_ready": False,
    }

    if src is None:
        raise RuntimeError("src 경로가 None 입니다.")
    src = Path(src)
    checks["src_exists"] = src.exists()
    if not checks["src_exists"]:
        raise FileNotFoundError(f"입력 영상 파일이 존재하지 않습니다: {src}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    checks["dest_parent_exists"] = dest.parent.exists()

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"입력 영상 열기 실패: {src}")
    checks["src_openable"] = True

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0 or fps <= 0 or W <= 0 or H <= 0:
        raise RuntimeError(
            f"영상 메타데이터 이상(total_frames={total_frames}, fps={fps}, size={W}x{H})"
        )
    checks["video_meta_ok"] = True

    if model_obj is None:
        raise RuntimeError("YOLO 모델 객체가 None 입니다.")
    if not hasattr(model_obj, "predict") and not callable(model_obj):
        raise RuntimeError("YOLO 모델 객체가 비정상입니다. predict/callable 확인 필요.")
    checks["model_ready"] = True

    writer = create_video_writer(dest, fps, W, H)
    ok = writer.isOpened()
    writer.release()
    if not ok:
        raise RuntimeError(f"출력 writer 초기화 실패: {dest} (avc1/mp4v 모두 실패)")
    checks["writer_openable"] = True

    return {
        "checks": checks,
        "fps": fps,
        "width": W,
        "height": H,
        "total_frames": total_frames,
    }

def angle_diff_deg(a: float, b: float) -> float:
    d = math.atan2(math.sin(a - b), math.cos(a - b))
    return abs(math.degrees(d))


# ──────────────────────────────────────────────────────────────
# Track
# ──────────────────────────────────────────────────────────────
class _Track:
    _next_id = 1

    def __init__(self, cx, cy, bbox, cls_id, frame_idx):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.yolo_id     = None
        self.trail       = deque(maxlen=TRAIL_LEN)
        self.trail.append((cx, cy))
        self.bbox        = bbox
        self.cls_id      = cls_id
        self.hits        = 1
        self.lost        = 0
        self.direction   = 'UNKNOWN'
        self.dir_score   = 0.0
        self.dir_label   = 'UNKNOWN'
        self.status      = 'NORMAL'
        self.stop_count  = 0
        # 역주행 확정 상태
        self.wrong_count           = 0
        self.wrong_confirmed       = False
        self.wrong_pending         = False
        self.wrong_last_seen_frame = frame_idx
        self.wrong_debug_reason    = 'init'
        self.wrong_debug_detail    = {}
        self.first_frame = frame_idx
        self.last_frame  = frame_idx
        self.full_trail  = [(cx, cy)]
        self.speeds      = []

    @property
    def cx(self): return self.trail[-1][0]
    @property
    def cy(self): return self.trail[-1][1]

    def update(self, cx, cy, bbox, cls_id, frame_idx):
        px, py = self.trail[-1]
        spd = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        self.speeds.append(spd)
        self.trail.append((cx, cy))
        self.full_trail.append((cx, cy))
        self.bbox       = bbox
        self.cls_id     = cls_id
        self.hits      += 1
        self.lost       = 0
        self.last_frame = frame_idx
        self.stop_count = self.stop_count + 1 if spd < STOP_SPEED_PX else max(0, self.stop_count - 2)

    def update_dir_score(self, raw_pred: str):
        target = +1.0 if raw_pred == 'UP' else -1.0 if raw_pred == 'DOWN' else 0.0
        self.dir_score = (1 - DIR_EMA_ALPHA) * self.dir_score + DIR_EMA_ALPHA * target
        if self.dir_score > DIR_SCORE_UP:
            self.dir_label = 'UP'
        elif self.dir_score < DIR_SCORE_DOWN:
            self.dir_label = 'DOWN'
        self.direction = self.dir_label

    def mean_angle(self):
        pts = list(self.trail)
        if len(pts) < MIN_TRAJ_LEN:
            return None
        h  = len(pts) // 2
        x0 = np.mean([p[0] for p in pts[:h]])
        y0 = np.mean([p[1] for p in pts[:h]])
        x1 = np.mean([p[0] for p in pts[h:]])
        y1 = np.mean([p[1] for p in pts[h:]])
        dx, dy = x1 - x0, y1 - y0
        return np.arctan2(dy, dx) if np.sqrt(dx**2 + dy**2) > 3 else None

    def avg_speed(self):
        return float(np.mean(self.speeds)) if self.speeds else 0.0

    def displacement(self):
        if len(self.full_trail) < 2:
            return 0.0
        x0, y0 = self.full_trail[0]
        x1, y1 = self.full_trail[-1]
        return float(np.sqrt((x1-x0)**2 + (y1-y0)**2))


# ──────────────────────────────────────────────────────────────
# 방향 분류기
# ──────────────────────────────────────────────────────────────
class _DirectionClassifier:
    def __init__(self):
        self.kmeans        = None
        self.up_cluster    = None
        self.prev_centers  = None
        self.road_axis     = None       # 부호 고정된 축: 양수 = DOWN 방향
        self._axis_locked  = False
        self._swap_count   = 0
        self._fit_count    = 0
        self.reason_stats  = {
            'kmeans_none': 0,
            'recent_vec_none': 0,
            'speed_low': 0,
            'disp_low': 0,
            'mean_angle_none': 0,
            'center_margin_fail': 0,
            'recent_not_opposite_enough': 0,
            'dir_unknown': 0,
            'same_as_zone_flow': 0,
            'out_of_zone': 0,
            'ok_candidate': 0,
        }
        self.up_zone_polygon = None
        self.down_zone_polygon = None

    def update_road_axis(self, tracks, frame_idx: int, fps: float = 30.0):
        vecs = []
        for t in tracks:
            a = t.mean_angle()
            if a is not None:
                vecs.append([np.cos(a), np.sin(a)])
        if len(vecs) < 2:
            return

        V = np.array(vecs, dtype=np.float32)
        try:
            _, _, Vt = np.linalg.svd(V, full_matrices=False)
        except np.linalg.LinAlgError:
            return
        new_axis = Vt[0]

        if self.road_axis is None:
            self.road_axis = new_axis
        else:
            if np.dot(new_axis, self.road_axis) < 0:
                new_axis = -new_axis
            raw  = (1 - FLOW_EMA_ALPHA) * self.road_axis + FLOW_EMA_ALPHA * new_axis
            norm = np.linalg.norm(raw)
            self.road_axis = raw / norm if norm > 1e-6 else self.road_axis

        axis_init_frames = int(fps * AXIS_INIT_S)
        if not self._axis_locked:
            if self.up_cluster is not None and self.kmeans is not None:
                up_center = self.kmeans.cluster_centers_[self.up_cluster]
                if np.dot(up_center, self.road_axis) > 0:
                    self.road_axis = -self.road_axis
                if frame_idx >= axis_init_frames:
                    self._axis_locked = True

    def fit(self, tracks) -> bool:
        valid_angles = [(t, t.mean_angle()) for t in tracks if t.mean_angle() is not None]
        if len(valid_angles) < MIN_VALID_TRACKS:
            return False

        X  = np.array([[np.cos(a), np.sin(a)] for _, a in valid_angles])
        km = KMeans(n_clusters=2, random_state=42, n_init=15)
        km.fit(X)

        new_c0 = km.cluster_centers_[0].copy()
        new_c1 = km.cluster_centers_[1].copy()

        a0   = np.arctan2(new_c0[1], new_c0[0])
        a1   = np.arctan2(new_c1[1], new_c1[0])
        diff = abs(np.degrees(np.arctan2(np.sin(a0 - a1), np.cos(a0 - a1))))
        if diff < 30:
            return False

        if self.prev_centers is not None:
            prev_c0, prev_c1 = self.prev_centers
            sim_00 = float(np.dot(new_c0, prev_c0))
            sim_01 = float(np.dot(new_c0, prev_c1))
            if sim_01 > sim_00:
                new_c0, new_c1 = new_c1, new_c0
                km.cluster_centers_[[0, 1]] = km.cluster_centers_[[1, 0]]
                self._swap_count += 1

            def _angle_shift(a, b):
                return abs(np.degrees(np.arctan2(float(np.cross(a, b)), float(np.dot(a, b)))))

            if (_angle_shift(prev_c0, new_c0) > SWAP_GUARD_DEG or
                    _angle_shift(prev_c1, new_c1) > SWAP_GUARD_DEG):
                return False

        self.kmeans       = km
        self.prev_centers = (new_c0.copy(), new_c1.copy())
        self._fit_count  += 1

        if self.up_cluster is None:
            self.up_cluster = 0 if new_c0[1] < new_c1[1] else 1

        return True

    def predict(self, track) -> str:
        if self.kmeans is None:
            return 'UNKNOWN'
        a = track.mean_angle()
        if a is None:
            return 'UNKNOWN'
        v   = np.array([np.cos(a), np.sin(a)])
        lbl = self.kmeans.predict([v])[0]
        return 'UP' if lbl == self.up_cluster else 'DOWN'

    def _recent_vec_unit(self, track):
        pts = list(track.trail)
        if len(pts) < WRONG_VEC_WIN:
            return None, 0.0, 0.0, 0.0

        sub = pts[-WRONG_VEC_WIN:]
        half = max(1, len(sub)//2)
        x0 = float(np.mean([p[0] for p in sub[:half]]))
        y0 = float(np.mean([p[1] for p in sub[:half]]))
        x1 = float(np.mean([p[0] for p in sub[half:]]))
        y1 = float(np.mean([p[1] for p in sub[half:]]))

        dx, dy = x1 - x0, y1 - y0
        disp = float(np.hypot(dx, dy))
        if disp < 1e-6:
            box_h = max(float(track.bbox[3] - track.bbox[1]), 1.0)
            disp_norm = disp / box_h
            return None, 0.0, disp, disp_norm

        v = np.array([dx, dy], dtype=np.float32) / disp
        avg_speed = track.avg_speed() if hasattr(track, "avg_speed") else 0.0
        box_h = max(float(track.bbox[3] - track.bbox[1]), 1.0)
        disp_norm = disp / box_h
        return v, avg_speed, disp, disp_norm

    def is_wrong_way(self, track) -> tuple[bool, str, dict]:
        """
        위치 기반 단순 wrong-way 판정
        - 파란 구역(상행) 안에서 DOWN이면 wrong-way
        - 녹색 구역(하행) 안에서 UP이면 wrong-way
        """
        if track.dir_label not in ('UP', 'DOWN'):
            self.reason_stats['dir_unknown'] += 1
            return False, 'dir_unknown', {'dir_label': track.dir_label}

        if self.up_zone_polygon is None or self.down_zone_polygon is None:
            self.reason_stats['kmeans_none'] += 1
            return False, 'zones_not_ready', {}

        v_recent, avg_speed, disp, disp_norm = self._recent_vec_unit(track)
        if v_recent is None:
            self.reason_stats['recent_vec_none'] += 1
            return False, 'recent_vec_none', {}

        if avg_speed < WRONG_MIN_SPEED:
            self.reason_stats['speed_low'] += 1
            return False, 'speed_low', {
                'avg_speed': round(float(avg_speed), 3),
                'need': WRONG_MIN_SPEED
            }

        if disp_norm < WRONG_MIN_DISP_RATIO:
            self.reason_stats['disp_low'] += 1
            return False, 'disp_low', {
                'disp': round(float(disp), 3),
                'disp_norm': round(float(disp_norm), 3),
                'need_ratio': WRONG_MIN_DISP_RATIO,
                'box_h': round(float(max(track.bbox[3] - track.bbox[1], 1)), 2),
            }

        cx, cy = track.cx, track.cy
        in_up_zone = cv2.pointPolygonTest(self.up_zone_polygon, (float(cx), float(cy)), False) >= 0
        in_down_zone = cv2.pointPolygonTest(self.down_zone_polygon, (float(cx), float(cy)), False) >= 0

        if not in_up_zone and not in_down_zone:
            self.reason_stats['out_of_zone'] += 1
            return False, 'out_of_zone', {'cx': cx, 'cy': cy}

        if in_up_zone and track.dir_label == 'DOWN':
            self.reason_stats['ok_candidate'] += 1
            return True, 'ok_candidate', {
                'zone': 'UP_ZONE',
                'track_dir': track.dir_label,
                'cx': cx, 'cy': cy,
                'disp_norm': round(float(disp_norm), 3),
                'avg_speed': round(float(avg_speed), 3),
            }

        if in_down_zone and track.dir_label == 'UP':
            self.reason_stats['ok_candidate'] += 1
            return True, 'ok_candidate', {
                'zone': 'DOWN_ZONE',
                'track_dir': track.dir_label,
                'cx': cx, 'cy': cy,
                'disp_norm': round(float(disp_norm), 3),
                'avg_speed': round(float(avg_speed), 3),
            }

        self.reason_stats['same_as_zone_flow'] += 1
        return False, 'same_as_zone_flow', {
            'zone': 'UP_ZONE' if in_up_zone else 'DOWN_ZONE',
            'track_dir': track.dir_label,
            'cx': cx, 'cy': cy
        }


    def build_flow_zones(self, tracks, frame_w: int, frame_h: int):
        """UP/DOWN 트랙 포인트들로 상행/하행 바닥 구역(사다리꼴) 생성"""
        up_pts = []
        down_pts = []

        for t in tracks:
            if getattr(t, 'hits', 0) < MIN_HITS:
                continue
            pts = list(getattr(t, 'full_trail', [])) or list(getattr(t, 'trail', []))
            if len(pts) < 6:
                continue
            if getattr(t, 'dir_label', None) == 'UP':
                up_pts.extend(pts)
            elif getattr(t, 'dir_label', None) == 'DOWN':
                down_pts.extend(pts)

        def _poly_from_points(points):
            if len(points) < 20:
                return None
            pts = np.array(points, dtype=np.int32)
            ys = pts[:, 1]

            y_top = int(np.percentile(ys, 15))
            y_bot = int(np.percentile(ys, 90))
            if y_bot - y_top < 30:
                return None

            top_band = pts[np.abs(pts[:, 1] - y_top) <= max(8, int((y_bot - y_top) * 0.07))]
            bot_band = pts[np.abs(pts[:, 1] - y_bot) <= max(10, int((y_bot - y_top) * 0.08))]
            if len(top_band) < 4 or len(bot_band) < 4:
                return None

            x_top_l = int(np.percentile(top_band[:, 0], 8))
            x_top_r = int(np.percentile(top_band[:, 0], 92))
            x_bot_l = int(np.percentile(bot_band[:, 0], 5))
            x_bot_r = int(np.percentile(bot_band[:, 0], 95))

            x_top_l = max(0, x_top_l - 8)
            x_top_r = min(frame_w - 1, x_top_r + 8)
            x_bot_l = max(0, x_bot_l - 15)
            x_bot_r = min(frame_w - 1, x_bot_r + 15)
            y_top = max(0, y_top - 4)
            y_bot = min(frame_h - 1, y_bot + 6)

            poly = np.array([
                [x_top_l, y_top],
                [x_top_r, y_top],
                [x_bot_r, y_bot],
                [x_bot_l, y_bot],
            ], dtype=np.int32)
            return poly

        self.up_zone_polygon = _poly_from_points(up_pts)
        self.down_zone_polygon = _poly_from_points(down_pts)

    def debug_info(self) -> dict:
        axis_deg = None
        if self.road_axis is not None:
            axis_deg = round(float(np.degrees(
                np.arctan2(self.road_axis[1], self.road_axis[0]))), 1)
        c0, c1 = None, None
        if self.kmeans is not None:
            c0 = round(np.degrees(np.arctan2(self.kmeans.cluster_centers_[0,1],
                                             self.kmeans.cluster_centers_[0,0])), 1)
            c1 = round(np.degrees(np.arctan2(self.kmeans.cluster_centers_[1,1],
                                             self.kmeans.cluster_centers_[1,0])), 1)
        return {
            "road_axis_deg": axis_deg,
            "axis_locked":   self._axis_locked,
            "center0_deg":   c0,
            "center1_deg":   c1,
            "up_cluster":    self.up_cluster,
            "swap_count":    self._swap_count,
            "fit_count":     self._fit_count,
            "reason_stats":  dict(self.reason_stats),
            "up_zone_ready": self.up_zone_polygon is not None,
            "down_zone_ready": self.down_zone_polygon is not None,
        }


# ──────────────────────────────────────────────────────────────
# 정체 판정
# ──────────────────────────────────────────────────────────────
def _is_congestion(track, all_valid_tracks, congestion_frames: int) -> bool:
    if track.stop_count < congestion_frames:
        return False
    cx, cy = track.cx, track.cy
    count = 0
    for other in all_valid_tracks:
        if other.id == track.id:
            continue
        if other.stop_count < congestion_frames:
            continue
        dist = np.sqrt((other.cx - cx)**2 + (other.cy - cy)**2)
        if dist <= CONGESTION_RADIUS:
            count += 1
    return count >= CONGESTION_MIN_N - 1


# ──────────────────────────────────────────────────────────────
# 이상 감지
# ──────────────────────────────────────────────────────────────
def _detect_anomaly(track, frame_w, frame_h, clf, valid_tracks, fps: float = 30.0):
    stop_frames       = max(1, int(fps * STOP_S))
    congestion_frames = max(1, int(fps * CONGESTION_S))

    is_wrong_now, wrong_reason, wrong_detail = clf.is_wrong_way(track)

    track.wrong_debug_reason = wrong_reason
    track.wrong_debug_detail = wrong_detail

    candidate_frames = max(1, int(fps * WRONG_CANDIDATE_S))
    confirm_frames   = max(1, int(fps * WRONG_CONFIRM_S))
    release_frames   = max(1, int(fps * WRONG_RELEASE_S))

    if is_wrong_now:
        track.wrong_count = min(track.wrong_count + 1, confirm_frames * 2)
        track.wrong_last_seen_frame = track.last_frame
    else:
        track.wrong_count = max(0, track.wrong_count - WRONG_DECAY)

    if (not track.wrong_confirmed) and track.wrong_count >= confirm_frames:
        track.wrong_confirmed = True
    if track.wrong_confirmed and track.wrong_count <= release_frames:
        track.wrong_confirmed = False

    track.wrong_pending = track.wrong_count >= candidate_frames

    wrong_confirmed = track.wrong_confirmed
    congestion_confirmed = False
    stop_confirmed = False

    if track.stop_count >= congestion_frames and track.hits >= MIN_HITS * 2:
        congestion_confirmed = _is_congestion(track, valid_tracks, congestion_frames)
        stop_confirmed = track.stop_count >= stop_frames

    if wrong_confirmed:
        return 'WRONG_WAY'
    if congestion_confirmed:
        return 'CONGESTION'
    if stop_confirmed:
        return 'STOPPED'
    return 'NORMAL'


# ──────────────────────────────────────────────────────────────
# 렌더링
# ──────────────────────────────────────────────────────────────
def _render_frame(frame, tracks, frame_idx, total_frames, clf, fps: float = 30.0):
    out = frame.copy()
    '''
    overlay = out.copy()
    if getattr(clf, 'up_zone_polygon', None) is not None:
        cv2.fillPoly(overlay, [clf.up_zone_polygon], COLOR_ZONE_UP)
    if getattr(clf, 'down_zone_polygon', None) is not None:
        cv2.fillPoly(overlay, [clf.down_zone_polygon], COLOR_ZONE_DOWN)
    out = cv2.addWeighted(overlay, ZONE_ALPHA, out, 1.0 - ZONE_ALPHA, 0)
    
    
    if getattr(clf, 'up_zone_polygon', None) is not None:
        cv2.polylines(out, [clf.up_zone_polygon], True, (255, 0, 0), 2)
        cv2.putText(out, "UP ZONE", tuple(clf.up_zone_polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 1, cv2.LINE_AA)
    if getattr(clf, 'down_zone_polygon', None) is not None:
        cv2.polylines(out, [clf.down_zone_polygon], True, (0, 180, 0), 2)
        cv2.putText(out, "DOWN ZONE", tuple(clf.down_zone_polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,180,0), 1, cv2.LINE_AA)
    '''

    for t in tracks:
        if t.hits < MIN_HITS:
            continue

        x1, y1, x2, y2 = t.bbox
        is_wrong    = t.status == 'WRONG_WAY'
        is_stop     = t.status == 'STOPPED'
        is_cong     = t.status == 'CONGESTION'

        if is_wrong or is_stop:
            color = COLOR_ANOMALY
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            label = f"[{ANOMALY_LABEL.get(t.status, t.status)}] #{t.id}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - lh - 10), (x1 + lw + 8, y1), color, -1)
            cv2.putText(out, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            if is_wrong and (frame_idx // 8) % 2 == 0:
                cv2.rectangle(out, (x1-4, y1-4), (x2+4, y2+4), (0, 0, 200), 2)

        elif is_cong:
            color = COLOR_CONGESTION
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"[CONG] #{t.id}"
            cv2.putText(out, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        else:
            color = COLOR_UP    if t.dir_label == 'UP'   else \
                    COLOR_DOWN  if t.dir_label == 'DOWN'  else COLOR_UNKNOWN
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            dir_txt = {'UP': 'UP', 'DOWN': 'DN', 'UNKNOWN': '?'}.get(t.dir_label, '?')
            cv2.putText(out, dir_txt, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            if t.wrong_count > 0:
                cf    = max(1, int(fps * WRONG_CONFIRM_S))
                ratio = min(1.0, t.wrong_count / cf)
                bw    = x2 - x1
                cv2.rectangle(out, (x1, y2+2), (x2, y2+6), (40,40,40), -1)
                cv2.rectangle(out, (x1, y2+2), (x1+int(bw*ratio), y2+6), (0,140,255), -1)

        trail  = list(t.trail)
        tcolor = COLOR_ANOMALY  if (is_wrong or is_stop) else \
                 COLOR_CONGESTION if is_cong else (
                 COLOR_UP    if t.dir_label == 'UP'  else
                 COLOR_DOWN  if t.dir_label == 'DOWN' else COLOR_UNKNOWN)
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            c  = tuple(int(ch * alpha) for ch in tcolor)
            p1 = (int(trail[i-1][0]), int(trail[i-1][1]))
            p2 = (int(trail[i][0]),   int(trail[i][1]))
            cv2.line(out, p1, p2, c, max(1, int(alpha * 2)), cv2.LINE_AA)

    dbg = clf.debug_info()
    panel_h = 255
    panel   = np.zeros((panel_h, 460, 3), np.uint8)
    panel[:] = (20, 20, 20)
    cv2.rectangle(panel, (0,0), (459, panel_h-1), (60,60,60), 1)

    up_n  = sum(1 for t in tracks if t.hits >= MIN_HITS and t.dir_label == 'UP'   and t.status == 'NORMAL')
    dn_n  = sum(1 for t in tracks if t.hits >= MIN_HITS and t.dir_label == 'DOWN' and t.status == 'NORMAL')
    st_n  = sum(1 for t in tracks if t.hits >= MIN_HITS and t.status == 'STOPPED')
    cg_n  = sum(1 for t in tracks if t.hits >= MIN_HITS and t.status == 'CONGESTION')
    ww_n  = sum(1 for t in tracks if t.hits >= MIN_HITS and t.status == 'WRONG_WAY')

    ax  = dbg['road_axis_deg']
    lck = 'Y' if dbg['axis_locked'] else 'N'
    c0d = dbg['center0_deg']
    c1d = dbg['center1_deg']
    uc  = dbg['up_cluster']
    swc = dbg['swap_count']

    cf = max(1, int(fps * WRONG_CONFIRM_S))
    ww_pending = [(t.id, t.wrong_count) for t in tracks
                  if t.hits >= MIN_HITS and t.wrong_count > 0 and not t.wrong_confirmed]
    ww_pend_str = ' '.join(f'#{tid}:{wc}/{cf}' for tid, wc in ww_pending[:3]) or '-'
    rs = dbg.get('reason_stats', {})
    reason_items = sorted([(k, v) for k, v in rs.items() if k != 'ok_candidate'], key=lambda x: x[1], reverse=True)
    top_reason = ', '.join([f"{k}:{v}" for k, v in reason_items[:3]]) if reason_items else '-'

    lines = [
        (f"Highway Anomaly  [{frame_idx}/{total_frames}]",           (200,200,200), 0.44),
        (f"  UP   : {up_n:2d}",                                      COLOR_UP,      0.50),
        (f"  DN   : {dn_n:2d}",                                      COLOR_DOWN,    0.50),
        (f"  STOPPED   : {st_n:2d}",                                 COLOR_ANOMALY, 0.46),
        (f"  CONGESTION: {cg_n:2d}",                                 COLOR_CONGESTION, 0.46),
        (f"  WRONG WAY : {ww_n:2d}",                                 COLOR_ANOMALY, 0.46),
        (f"  AXIS={ax}deg(lck={lck})  C0={c0d} C1={c1d} UP={uc}", (160,200,160), 0.36),
        (f"  SwapCnt={swc}  WW_pend={ww_pend_str}",                  (160,160,200), 0.36),
        (f"  WW_false_top: {top_reason}",                               (200,180,120), 0.32),
    ]
    for i, (txt, col, sc) in enumerate(lines):
        cv2.putText(panel, txt, (8, 20 + i*24),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
    out[10:10+panel_h, 10:470] = panel

    return out


# ──────────────────────────────────────────────────────────────
# VideoDetectorHighway
# ──────────────────────────────────────────────────────────────
class VideoDetectorHighway:
    """고속도로 이상 주행 감지기."""

    def __init__(self):
        from ultralytics import YOLO
        model_path = resolve_model_path(
            "yolov8n.pt",
            "yolo11n.pt",
            purpose="VideoDetectorHighway",
        )
        self.model = YOLO(str(model_path), verbose=False)

    def detect(self, src: Path, dest: Path, **kwargs) -> dict:
        _Track._next_id = 1
        t_start = time.time()
        stage = "init"
        progress_every = int(kwargs.get("progress_every", 60))
        writer = None

        try:
            stage = "precheck"
            _stage_print(stage, f"checking source={src} dest={dest}")
            pre = _precheck_before_detect(Path(src), Path(dest), self.model)
            fps = float(pre["fps"])
            W = int(pre["width"])
            H = int(pre["height"])
            total_frames = int(pre["total_frames"])
            _stage_print(stage, f"ok checks={pre['checks']} meta={{'frames': {total_frames}, 'fps': {fps:.3f}, 'size': '{W}x{H}'}}")

            stage = "init_writer"
            fourcc = cv2.VideoWriter.fourcc(*"avc1")
            writer = cv2.VideoWriter(str(dest), fourcc, fps, (W, H))
            if not writer.isOpened():
                writer.release()
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(dest), fourcc, fps, (W, H))
            if not writer.isOpened():
                raise RuntimeError(f"출력 writer 초기화 실패: {dest}")

            learn_frames = min(total_frames, max(1, int(fps * 30.0)))
            _stage_print(stage, f"learn_frames={learn_frames} ({learn_frames/fps:.2f}s)")

            def _run_tracker_pass(cap_obj, frame_limit=None, clf_obj=None, do_learn=True, do_render=False, pass_name="pass"):
                tracks = []
                all_tracks = []
                # YOLO track id -> _Track
                yolo_track_map = {}
                frame_idx = 0
                det_total = 0

                while True:
                    ret, frame = cap_obj.read()
                    if not ret:
                        break
                    if frame_limit is not None and frame_idx >= frame_limit:
                        break

                    # YOLO 내부 tracker 사용 (persist=True)
                    results = self.model.track(
                        frame,
                        conf=CONF_THRESH,
                        classes=VEHICLE_CLASSES,
                        verbose=False,
                        persist=TRACK_PERSIST,
                        tracker=TRACKER_CFG,
                    )

                    detections = []
                    for r in results:
                        if r.boxes is None:
                            continue
                        ids = None
                        if getattr(r.boxes, "id", None) is not None:
                            try:
                                ids = r.boxes.id.int().cpu().tolist()
                            except Exception:
                                ids = None

                        xyxy_list = r.boxes.xyxy.int().cpu().tolist() if getattr(r.boxes, "xyxy", None) is not None else []
                        cls_list = r.boxes.cls.int().cpu().tolist() if getattr(r.boxes, "cls", None) is not None else []

                        for i, xyxy in enumerate(xyxy_list):
                            x1, y1, x2, y2 = map(int, xyxy)
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            cls_id = int(cls_list[i]) if i < len(cls_list) else -1
                            yolo_id = int(ids[i]) if ids is not None and i < len(ids) and ids[i] is not None else None
                            detections.append((yolo_id, cx, cy, (x1, y1, x2, y2), cls_id))
                    det_total += len(detections)

                    updated_tracks = set()
                    unmatched = []

                    # 1) YOLO track id가 있는 detection은 그 id로 직접 업데이트
                    for yolo_id, cx, cy, bbox, cls_id in detections:
                        if yolo_id is None:
                            unmatched.append((yolo_id, cx, cy, bbox, cls_id))
                            continue

                        if yolo_id in yolo_track_map and yolo_track_map[yolo_id] in tracks:
                            t = yolo_track_map[yolo_id]
                            t.update(cx, cy, bbox, cls_id, frame_idx)
                        else:
                            nt = _Track(cx, cy, bbox, cls_id, frame_idx)
                            nt.yolo_id = yolo_id
                            tracks.append(nt)
                            all_tracks.append(nt)
                            yolo_track_map[yolo_id] = nt
                            t = nt
                        updated_tracks.add(id(t))

                    # 2) id 없는 detection만 기존 최근접 매칭을 보조적으로 사용
                    if unmatched:
                        matched_unmatched = set()
                        candidate_tracks = [t for t in tracks if id(t) not in updated_tracks]

                        for t in candidate_tracks:
                            best_d, best_i = float('inf'), -1
                            for i, (_, cx, cy, bbox, cls_id) in enumerate(unmatched):
                                if i in matched_unmatched:
                                    continue
                                d = (t.cx - cx) ** 2 + (t.cy - cy) ** 2
                                if d < best_d:
                                    best_d, best_i = d, i
                            if best_d < MAX_DIST ** 2 and best_i >= 0:
                                _, cx, cy, bbox, cls_id = unmatched[best_i]
                                t.update(cx, cy, bbox, cls_id, frame_idx)
                                matched_unmatched.add(best_i)
                                updated_tracks.add(id(t))

                        for i, (_, cx, cy, bbox, cls_id) in enumerate(unmatched):
                            if i not in matched_unmatched:
                                nt = _Track(cx, cy, bbox, cls_id, frame_idx)
                                tracks.append(nt)
                                all_tracks.append(nt)
                                updated_tracks.add(id(nt))

                    # 3) 이번 프레임에 업데이트 안 된 track은 lost 증가
                    for t in tracks:
                        if id(t) not in updated_tracks:
                            t.lost += 1

                    # 4) MAX_LOST 초과 track 제거 + yolo_track_map 정리
                    kept = []
                    for t in tracks:
                        if t.lost <= MAX_LOST:
                            kept.append(t)
                        else:
                            if getattr(t, "yolo_id", None) is not None:
                                cur = yolo_track_map.get(t.yolo_id)
                                if cur is t:
                                    del yolo_track_map[t.yolo_id]
                    tracks = kept

                    valid_tracks = [t for t in tracks if t.hits >= MIN_HITS]

                    if do_learn:
                        clf_obj.update_road_axis(valid_tracks, frame_idx, fps)
                        if frame_idx % CLUSTER_EVERY == 0:
                            clf_obj.fit(valid_tracks)

                    for t in valid_tracks:
                        t.update_dir_score(clf_obj.predict(t))
                    for t in valid_tracks:
                        t.status = _detect_anomaly(t, W, H, clf_obj, valid_tracks, fps)

                    if do_render:
                        writer.write(_render_frame(frame, tracks, frame_idx, total_frames, clf_obj, fps))

                    if frame_idx == 0 or ((frame_idx + 1) % progress_every == 0):
                        dbg = clf_obj.debug_info()
                        wrong_n = sum(1 for t in valid_tracks if t.status == 'WRONG_WAY')
                        stop_n = sum(1 for t in valid_tracks if t.status == 'STOPPED')
                        cong_n = sum(1 for t in valid_tracks if t.status == 'CONGESTION')
                        rs = dbg.get('reason_stats', {})
                        reason_items = sorted([(k, v) for k, v in rs.items() if k != 'ok_candidate'], key=lambda x: x[1], reverse=True)
                        top_false = reason_items[0][0] if reason_items else '-'
                        yolo_live = sum(1 for t in tracks if getattr(t, 'yolo_id', None) is not None)
                        _stage_print(
                            pass_name,
                            f"frame={frame_idx+1}/{frame_limit if frame_limit is not None else total_frames} "
                            f"tracks={len(tracks)} yolo_live={yolo_live} valid={len(valid_tracks)} det_total={det_total} "
                            f"fit={dbg['fit_count']} axis={dbg['road_axis_deg']} locked={dbg['axis_locked']} "
                            f"ww={wrong_n} stop={stop_n} cong={cong_n} top_false={top_false}"
                        )

                    frame_idx += 1

                for t in tracks:
                    if t not in all_tracks:
                        all_tracks.append(t)
                dbg = clf_obj.debug_info()
                _stage_print(
                    pass_name,
                    f"done frames={frame_idx}, total_tracks={len(all_tracks)}, fit={dbg['fit_count']}, "
                    f"axis={dbg['road_axis_deg']}, locked={dbg['axis_locked']}"
                )
                return all_tracks, clf_obj

            stage = "pass1_learn"
            _stage_print(stage, "start 30s prelearn")
            clf_pre = _DirectionClassifier()
            cap1 = cv2.VideoCapture(str(src))
            if not cap1.isOpened():
                raise RuntimeError(f"PASS1 영상 열기 실패: {src}")
            tracks1, clf_pre = _run_tracker_pass(cap1, frame_limit=learn_frames, clf_obj=clf_pre, do_learn=True, do_render=False, pass_name="PASS1")
            cap1.release()
            clf_pre.build_flow_zones(tracks1, W, H)

            if clf_pre.kmeans is None:
                raise RuntimeError("PASS1 학습 실패: KMeans 미생성 (유효 트랙 부족 또는 방향 학습 실패)")
            if clf_pre.road_axis is None:
                raise RuntimeError("PASS1 학습 실패: road_axis 미생성 (방향축 추정 실패)")

            stage = "freeze_model"
            clf = _DirectionClassifier()
            clf.kmeans = clf_pre.kmeans
            clf.up_cluster = clf_pre.up_cluster
            clf.prev_centers = clf_pre.prev_centers
            clf.road_axis = None if clf_pre.road_axis is None else clf_pre.road_axis.copy()
            clf._axis_locked = True if clf.road_axis is not None else False
            clf._swap_count = clf_pre._swap_count
            clf._fit_count = clf_pre._fit_count
            clf.up_zone_polygon = None if clf_pre.up_zone_polygon is None else clf_pre.up_zone_polygon.copy()
            clf.down_zone_polygon = None if clf_pre.down_zone_polygon is None else clf_pre.down_zone_polygon.copy()
            _stage_print(stage, f"frozen axis={clf.debug_info()['road_axis_deg']} up_cluster={clf.up_cluster} fit={clf._fit_count} up_zone={clf.debug_info()['up_zone_ready']} down_zone={clf.debug_info()['down_zone_ready']}")

            stage = "pass2_detect"
            _stage_print(stage, "start full replay detect/render")
            cap2 = cv2.VideoCapture(str(src))
            if not cap2.isOpened():
                raise RuntimeError(f"PASS2 영상 열기 실패: {src}")
            all_tracks, clf = _run_tracker_pass(cap2, frame_limit=None, clf_obj=clf, do_learn=False, do_render=True, pass_name="PASS2")
            cap2.release()
            writer.release()

            stage = "build_results"
            dbg = clf.debug_info()
            _stage_print(stage, f"road_axis={dbg['road_axis_deg']}deg locked={dbg['axis_locked']} C0={dbg['center0_deg']} C1={dbg['center1_deg']} up_cluster={dbg['up_cluster']} swap={dbg['swap_count']} fit={dbg['fit_count']}")

            vehicles  = []
            anomalies = []

            for t in all_tracks:
                if t.hits < MIN_HITS:
                    continue
                traj  = [[round(x,1), round(y,1)] for i,(x,y) in enumerate(t.full_trail) if i%5==0]
                entry = {
                    "id": t.id,
                    "class": CLS_NAME.get(t.cls_id, 'vehicle'),
                    "direction": t.dir_label,
                    "status": t.status,
                    "status_ko": {'NORMAL':'정상','STOPPED':'정차','WRONG_WAY':'역주행','CONGESTION':'정체'}.get(t.status),
                    "first_frame": t.first_frame,
                    "last_frame": t.last_frame,
                    "total_frames": t.hits,
                    "avg_speed_px": round(t.avg_speed(), 2),
                    "displacement_px": round(t.displacement(), 1),
                    "mean_angle_deg": round(np.degrees(t.mean_angle()), 1) if t.mean_angle() is not None else None,
                    "trajectory": traj,
                }
                vehicles.append(entry)

                if t.status not in ('NORMAL', 'CONGESTION'):
                    anomalies.append({
                        "vehicle_id": t.id,
                        "type": t.status,
                        "type_ko": entry["status_ko"],
                        "class": entry["class"],
                        "direction": t.dir_label,
                        "position": [round(t.cx,1), round(t.cy,1)],
                        "frame": t.last_frame,
                        "timestamp_sec": round(t.last_frame / fps, 2),
                    })

            summary = {
                "status": "ok",
                "total_vehicles": len(vehicles),
                "upward": sum(1 for v in vehicles if v['direction'] == 'UP'),
                "downward": sum(1 for v in vehicles if v['direction'] == 'DOWN'),
                "unknown": sum(1 for v in vehicles if v['direction'] == 'UNKNOWN'),
                "stopped": sum(1 for v in vehicles if v['status'] == 'STOPPED'),
                "congestion": sum(1 for v in vehicles if v['status'] == 'CONGESTION'),
                "wrong_way": sum(1 for v in vehicles if v['status'] == 'WRONG_WAY'),
                "anomaly_count": len(anomalies),
                "process_time_sec": round(time.time() - t_start, 1),
                "debug": clf.debug_info(),
                "learn_frames": learn_frames,
                "learn_seconds": round(learn_frames / fps, 2),
            }

            _stage_print("complete", f"done process_time={summary['process_time_sec']}s vehicles={summary['total_vehicles']} wrong={summary['wrong_way']} stop={summary['stopped']} cong={summary['congestion']}")
            return {
                "video_info": {
                    "filename": str(src), "width": W, "height": H,
                    "fps": fps, "total_frames": total_frames,
                    "duration_sec": round(total_frames/fps, 2),
                },
                "summary": summary,
                "vehicles": vehicles,
                "anomalies": anomalies,
            }

        except Exception as e:
            err_payload = {
                "status": "error",
                "stage": stage,
                "error_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "src": str(src),
                "dest": str(dest),
                "elapsed_sec": round(time.time() - t_start, 2),
            }
            _stage_print("ERROR", f"stage={stage} type={type(e).__name__} message={e}")
            _write_error_sidecar(dest, err_payload)
            raise RuntimeError(f"detect 실패(stage={stage}): {type(e).__name__}: {e}") from e
