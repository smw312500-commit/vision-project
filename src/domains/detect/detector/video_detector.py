from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.domains.detect.detector.utils import (
    create_video_writer,
    debug_print,
    open_video_capture,
    resolve_model_path,
)


class BaseVideoDetector:
    def __init__(self, model_src: Path, **kwargs):
        self.model = YOLO(str(model_src), **kwargs)

    def detect(self, src: Path, dest: Path, **kwargs):
        cap, fps, width, height = open_video_capture(src)
        out = create_video_writer(dest, fps, width, height)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, **kwargs)
                out.write(results[0].plot())
        finally:
            out.release()
            cap.release()


class VideoDetectorYolo11n(BaseVideoDetector):
    def __init__(self):
        super().__init__(
            resolve_model_path("yolo11n.pt", purpose="VideoDetectorYolo11n"),
            verbose=False,
        )


class VideoDetectorFireDetectV1(BaseVideoDetector):
    def __init__(self):
        super().__init__(
            resolve_model_path(
                "fire_detect_v251205_1.pt", purpose="VideoDetectorFireDetectV1"
            ),
            verbose=False,
        )


# 갓길 정차 차량 감지 로직
class VideoDetectorShoulderStop:
    def __init__(self):
        self.model = YOLO(
            str(
                resolve_model_path(
                    "yolov8s.pt",
                    "yolo11n.pt",
                    purpose="VideoDetectorShoulderStop",
                )
            ),
            verbose=False,
        )

        self.STOP_DISTANCE = 3
        self.STOP_TIME_SECONDS = 2.0
        self.SKIP = 4

        # 고정좌표
        self.poly = np.array(
            [
                (459, 359),
                (519, 364),
                (283, 580),
                (110, 544),
            ],
            dtype=np.int32,
        )

        self._reset_state()

    def _reset_state(self):
        self.car_status = {}
        self.stopped_ids = set()
        self.stop_time_map = {}
        self.frame_count = 0
        self.last_boxes = []
        self.last_ids = []
        self.last_roi_occupied = False

    def detect(self, src: Path, dest: Path, **kwargs):
        self._reset_state()
        cap, fps, width, height = open_video_capture(src)
        out = create_video_writer(dest, fps, width, height)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            if self.frame_count % self.SKIP != 0:
                annotated = frame.copy()

                for box, tid in zip(self.last_boxes, self.last_ids):
                    x1, y1, x2, y2 = map(int, box)

                    color = (0, 0, 255) if tid in self.stopped_ids else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ID:{tid}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                roi_color = (0, 0, 255) if self.last_roi_occupied else (0, 255, 0)
                cv2.polylines(annotated, [self.poly], True, roi_color, 2)

                out.write(annotated)
                continue

            annotated = frame.copy()
            current_video_time = self.frame_count / fps
            roi_occupied = False

            results = self.model.track(
                frame,
                persist=True,
                conf=0.05,
                verbose=False,
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                ids = results[0].boxes.id.int().cpu().tolist()

                self.last_boxes = boxes
                self.last_ids = ids

                for box, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    corners = [
                        (x1, y1),
                        (x2, y1),
                        (x2, y2),
                        (x1, y2),
                    ]

                    inside = any(
                        cv2.pointPolygonTest(self.poly, corner, False) >= 0
                        for corner in corners
                    )

                    if not inside:
                        continue

                    roi_occupied = True

                    # 정차 판단 부분
                    if tid not in self.car_status:
                        self.car_status[tid] = (cx, cy, current_video_time)
                    else:
                        px, py, last_seen_time = self.car_status[tid]
                        move_dist = abs(cx - px) + abs(cy - py)

                        if move_dist < self.STOP_DISTANCE:
                            if (
                                current_video_time - last_seen_time
                                >= self.STOP_TIME_SECONDS
                            ):
                                self.stopped_ids.add(tid)
                                self.stop_time_map.setdefault(tid, current_video_time)
                        else:
                            self.car_status[tid] = (cx, cy, current_video_time)
                            self.stopped_ids.discard(tid)
                            self.stop_time_map.pop(tid, None)

                    color = (0, 0, 255) if tid in self.stopped_ids else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ID:{tid}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            self.last_roi_occupied = roi_occupied

            roi_color = (0, 0, 255) if roi_occupied else (0, 255, 0)
            cv2.polylines(annotated, [self.poly], True, roi_color, 2)
            out.write(annotated)

        cap.release()
        out.release()

class VideoDetectorWrongWay(BaseVideoDetector):
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorbike, bus, truck (COCO)

    # ✅ 탐지/트래킹 강화 상수는 일단 유지해도 되지만, yaml은 제거(미사용)
    DETECT_IMGSZ = 1280
    DETECT_CONF = 0.15
    DETECT_IOU = 0.55

    def __init__(self):
        super().__init__(
            resolve_model_path("yolo11n.pt", purpose="VideoDetectorWrongWay"),
            verbose=False,
        )

    def _get_centers(self, result, allowed_classes=None):
        centers = {}
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if allowed_classes is not None and cls_id not in allowed_classes:
                continue
            if box.id is None:
                continue
            track_id = int(box.id)
            x1, y1, x2, y2 = box.xyxy[0]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            centers[track_id] = (cx, cy)
        return centers

    def _get_tracks(self, result, allowed_classes=None):
        tracks = {}
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if allowed_classes is not None and cls_id not in allowed_classes:
                continue
            if box.id is None:
                continue
            track_id = int(box.id)
            x1, y1, x2, y2 = box.xyxy[0]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            tracks[track_id] = (cx, cy, bw, bh)
        return tracks

    def _lane_direction_step(
        self,
        track_id,
        move_dir,
        lane_map,
        lane_acc,
        normal_up,
        normal_down,
        cnt_thresh=10,
    ):
        debug_print("_lane_direction_step" + "=" * 40)
        if track_id in lane_map:
            return lane_map[track_id]

        acc_vec, cnt = lane_acc.get(track_id, (np.zeros(2, dtype=np.float32), 0))
        acc_vec += move_dir
        cnt += 1
        lane_acc[track_id] = (acc_vec, cnt)

        if cnt >= cnt_thresh:
            norm = np.linalg.norm(acc_vec)
            if norm < 1e-6:
                return None

            avg_vec = acc_vec / norm
            scores = {}
            if normal_up is not None:
                scores["up"] = float(np.dot(avg_vec, normal_up))
            if normal_down is not None:
                scores["down"] = float(np.dot(avg_vec, normal_down))

            if not scores:
                return None

            lane = max(scores, key=scores.get)
            lane_map[track_id] = lane
            lane_acc.pop(track_id, None)
            return lane

        return None

    def _learn_direction(
        self,
        video_path: str,
        warmup_seconds=45,
        max_match_dist=80.0,
        min_step=1.0,
        refine_iters=2,
        hysteresis_H=0.15,
    ):
        debug_print("_learn_direction" + "=" * 40)
        cap, fps, _, _ = open_video_capture(video_path)
        warmup_frames = int(fps * float(warmup_seconds))

        frame_count = 0
        prev_centers = None
        vectors = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count > warmup_frames:
                break

            # ✅ 원복: tracker/imgsz/conf/iou 제거
            result = self.model.track(
                frame,
                persist=True,
                verbose=False,
                classes=self.VEHICLE_CLASSES,
                imgsz=self.DETECT_IMGSZ,
                conf=self.DETECT_CONF,
                iou=self.DETECT_IOU,
            )[0]

            centers_dict = self._get_centers(
                result, allowed_classes=self.VEHICLE_CLASSES
            )
            if len(centers_dict) == 0:
                prev_centers = None
                continue

            centers = np.array(list(centers_dict.values()), dtype=np.float32)

            if prev_centers is not None:
                for prev in prev_centers:
                    dists = np.linalg.norm(centers - prev, axis=1)
                    j = int(np.argmin(dists))
                    if dists[j] < max_match_dist:
                        dx, dy = centers[j] - prev
                        step = float(np.hypot(dx, dy))
                        if step < float(min_step):
                            continue
                        vectors.append([dx, dy])

            prev_centers = centers

        cap.release()
        if len(vectors) == 0:
            return None, None

        vecs = np.stack(vectors, axis=0).astype(np.float32)
        angles = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
        median_angle = float(np.median(angles))
        labels = np.zeros(len(vecs), dtype=np.int32)  # 0=up, 1=down
        labels[angles > median_angle] = 1

        def compute_normal(group_vecs):
            if len(group_vecs) == 0:
                return None
            m = np.mean(group_vecs, axis=0)
            n = float(np.linalg.norm(m))
            if n < 1e-6:
                return None
            return (m / n).astype(np.float32)

        normal_up = compute_normal(vecs[labels == 0])
        normal_down = compute_normal(vecs[labels == 1])
        if normal_up is None or normal_down is None:
            return normal_up, normal_down

        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        uvecs = vecs / norms

        for _ in range(int(refine_iters)):
            up_scores = np.dot(uvecs, normal_up)
            down_scores = np.dot(uvecs, normal_down)
            diff = up_scores - down_scores

            new_labels = labels.copy()
            new_labels[diff > float(hysteresis_H)] = 0
            new_labels[diff < -float(hysteresis_H)] = 1
            labels = new_labels

            new_up = compute_normal(uvecs[labels == 0])
            new_down = compute_normal(uvecs[labels == 1])
            if new_up is None or new_down is None:
                break
            normal_up, normal_down = new_up, new_down

        return normal_up, normal_down

    def _build_lane_masks(
        self,
        video_path: str,
        normal_up,
        normal_down,
        paint_seconds=45,
        min_step=1.0,
        cnt_thresh=10,
        thickness_scale=0.95,
        thickness_min=9,
        thickness_max=45,
    ):
        debug_print("_build_lane_masks" + "=" * 40)
        cap, fps, _, _ = open_video_capture(video_path)

        paint_frames = int(fps * float(paint_seconds))

        # ✅ 원본 길이 유지
        frame_step = 1

        ret, frame0 = cap.read()
        if not ret:
            cap.release()
            return None, None

        H, W = frame0.shape[:2]
        mask_up = np.zeros((H, W), dtype=np.uint8)
        mask_down = np.zeros((H, W), dtype=np.uint8)

        prev = {}
        lane_map = {}
        lane_acc = {}
        life_cnt = {}

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if (frame_idx % frame_step) != 0:
                continue
            if frame_idx > paint_frames:
                break

            # ✅ 원복: tracker/imgsz/conf/iou 제거
            result = self.model.track(
                frame,
                persist=True,
                verbose=False,
                classes=self.VEHICLE_CLASSES,
                imgsz=self.DETECT_IMGSZ,
                conf=self.DETECT_CONF,
                iou=self.DETECT_IOU,
            )[0]

            tracks = self._get_tracks(result, allowed_classes=self.VEHICLE_CLASSES)

            for tid, (cx, cy, bw, bh) in tracks.items():
                if tid not in prev:
                    prev[tid] = (cx, cy)
                    life_cnt[tid] = 1
                    continue

                px, py = prev[tid]
                dx, dy = cx - px, cy - py
                prev[tid] = (cx, cy)

                step = float(np.hypot(dx, dy))
                if step < float(min_step):
                    continue

                base = max(bw, bh)
                if base < 18:
                    continue
                step_max = max(120.0, min(360.0, base * 1.6))
                if step > float(step_max):
                    continue

                life_cnt[tid] += 1
                if life_cnt[tid] < 5:
                    continue

                move_dir = np.array([dx, dy], dtype=np.float32) / (step + 1e-12)

                lane = self._lane_direction_step(
                    tid,
                    move_dir,
                    lane_map,
                    lane_acc,
                    normal_up,
                    normal_down,
                    cnt_thresh=cnt_thresh,
                )
                if lane is None:
                    continue

                thickness = int(base * thickness_scale)
                thickness = max(thickness_min, min(thickness_max, thickness))

                p1 = (int(px), int(py))
                p2 = (int(cx), int(cy))

                if lane == "up":
                    cv2.line(mask_up, p1, p2, 255, thickness, cv2.LINE_AA)
                else:
                    cv2.line(mask_down, p1, p2, 255, thickness, cv2.LINE_AA)

        cap.release()
        return mask_up, mask_down

    def detect(self, src: Path, dest: Path, **kwargs):
        video_path = str(src)
        alerts = []
        wrong_hold = {}
        last_lane = {}

        cap, fps, width, height = open_video_capture(src)
        frame_step = 1

        HOLD_SECONDS = 4.0
        HOLD_FRAMES = int(fps * HOLD_SECONDS)

        debug_print("before learn direction")
        normal_up, normal_down = self._learn_direction(video_path)
        if normal_up is None or normal_down is None:
            cap.release()
            raise RuntimeError("learn_direction failed (not enough motion vectors)")

        debug_print("before build lane masks")
        mask_up, mask_down = self._build_lane_masks(video_path, normal_up, normal_down)
        if mask_up is None or mask_down is None:
            cap.release()
            raise RuntimeError("build_lane_masks failed")

        out = create_video_writer(dest, fps, width, height)

        threshold_frames = int(fps * 0.12)
        min_step = 1.0
        dot_threshold = 0.03

        prev_centers = {}
        wrong_counter = {}

        debug_print("main loop")
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if (frame_idx % frame_step) != 0:
                    continue

                if wrong_hold:
                    for tid in list(wrong_hold.keys()):
                        wrong_hold[tid] -= 1
                        if wrong_hold[tid] <= 0:
                            wrong_hold.pop(tid, None)

                # ✅ 원복: tracker/imgsz/conf/iou 제거
                result = self.model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    classes=self.VEHICLE_CLASSES,
                    imgsz=self.DETECT_IMGSZ,
                    conf=self.DETECT_CONF,
                    iou=self.DETECT_IOU,
                )[0]

                box_map = {}
                for b in result.boxes:
                    if b.id is None:
                        continue
                    tid = int(b.id)
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    box_map[tid] = (x1, y1, x2, y2)

                annotated = frame.copy()
                centers = self._get_centers(
                    result, allowed_classes=self.VEHICLE_CLASSES
                )
                # 벡터 표시
                """
                if mask_up is not None and mask_down is not None:
                    # up=green, down=blue
                    g = mask_up
                    b = mask_down
                    annotated[:, :, 1] = np.maximum(annotated[:, :, 1], g)
                    annotated[:, :, 0] = np.maximum(annotated[:, :, 0], b)
                """

                for track_id, (cx, cy) in centers.items():
                    if track_id not in prev_centers:
                        prev_centers[track_id] = (cx, cy)
                        wrong_counter[track_id] = 0
                        continue

                    prev_cx, prev_cy = prev_centers[track_id]
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    prev_centers[track_id] = (cx, cy)

                    step = float((dx * dx + dy * dy) ** 0.5)
                    if step < min_step:
                        continue

                    move_dir = np.array([dx, dy], dtype=np.float32) / (step + 1e-12)

                    cx_i, cy_i = int(cx), int(cy)
                    r = 12
                    y1 = max(0, cy_i - r)
                    y2 = min(mask_up.shape[0], cy_i + r + 1)
                    x1 = max(0, cx_i - r)
                    x2 = min(mask_up.shape[1], cx_i + r + 1)

                    up_score = int(np.sum(mask_up[y1:y2, x1:x2]))
                    down_score = int(np.sum(mask_down[y1:y2, x1:x2]))

                    LANE_MARGIN = 200
                    if up_score - down_score > LANE_MARGIN:
                        expected_normal = normal_up
                        lane_name = "up"
                        last_lane[track_id] = lane_name
                    elif down_score - up_score > LANE_MARGIN:
                        expected_normal = normal_down
                        lane_name = "down"
                        last_lane[track_id] = lane_name
                    else:
                        lane_name = last_lane.get(track_id)
                        if lane_name is None:
                            lane_name = "up" if (up_score >= down_score) else "down"
                            last_lane[track_id] = lane_name
                        expected_normal = (
                            normal_up if lane_name == "up" else normal_down
                        )

                    sim = float(np.dot(move_dir, expected_normal))
                    if sim < -dot_threshold:
                        wrong_counter[track_id] += 1
                    else:
                        wrong_counter[track_id] = 0

                    if wrong_counter[track_id] > threshold_frames:
                        wrong_hold[track_id] = HOLD_FRAMES
                        alerts.append(
                            {
                                "type": "WRONG_WAY",
                                "track_id": track_id,
                                "lane": lane_name,
                                "frame": frame_idx,
                            }
                        )
                        wrong_counter[track_id] = 0

                    if track_id in box_map:
                        bx1, by1, bx2, by2 = box_map[track_id]
                        cv2.putText(
                            annotated,
                            f"ID:{track_id} {lane_name}",
                            (bx1, max(20, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )

                    if wrong_hold.get(track_id, 0) > 0 and track_id in box_map:
                        rx1, ry1, rx2, ry2 = box_map[track_id]
                        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                        cv2.putText(
                            annotated,
                            "Wrong Way",
                            (rx1, max(20, ry1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            2,
                        )

                out.write(annotated)

        finally:
            out.release()
            cap.release()

        return {"output_video": str(dest), "alerts": alerts}
