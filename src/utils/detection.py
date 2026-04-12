from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger
from ultralytics.models import YOLO

# CONSTANTS
PERSON_CLASS_ID = 0
LABEL_MAP = {0: "smoke", 1: "fire"}

# PERSON DETECTION (WITH TRACKING)
def person_detection(
    frame: np.ndarray,
    model: YOLO,
    thresh: float
) -> Dict[str, Any]:
    """
    Person Detection using YOLO Model trained of COCO Dataset

    Args:
        - frame (np.ndarray): Image / Video Frame to be inferred
        - model (YOLO): YOLO model
        - thresh (float): Threshold confidence for person detection

    Returns:
        - returns Dict[str, Any] which is dictionary of detection information.
    """

    detections = []
    detect_person = False

    res = model.track(frame, conf=thresh, iou=0.6, persist=True)[0]

    if res.boxes is not None:
        for box in res.boxes:
            cls = int(box.cls[0])

            if cls == PERSON_CLASS_ID:
                detect_person = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                track_id = (
                    int(box.id[0])
                    if (box.id is not None and len(box.id) > 0)
                    else -1
                )

                detections.append({
                    "type": "person",
                    "id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf
                })

    return {
        "flags": {"person": detect_person},
        "detections": detections
    }


# FIRE + SMOKE DETECTION (YOLO)
def fire_smoke_detection(
    frame: np.ndarray,
    model: YOLO,
    thresh: float
) -> Dict[str, Any]:
    """
    Fire and Smoke Detection using YOLO Model trained on custom fire and smoke
    dataset.

    Args:
        - frame (np.ndarray): Image / Video Frame to be inferred
        - model (YOLO): YOLO model (custom model)
        - thresh (float): Threshold confidence for person detection

    Returns:
        - returns Dict[str, Any] which is dictionary containing detection
        information of fire and smoke.
    """

    detections: List[Dict[str, Any]] = []
    detect_fire = False
    detect_smoke = False

    res = model(frame, conf=thresh, iou=0.6, classes=[0, 1])[0]

    if res.boxes is not None:
        for box in res.boxes:
            cls = int(box.cls[0])
            label = LABEL_MAP.get(cls)

            if label:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                detections.append({
                    "type": label,   # "fire" or "smoke"
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf
                })

                if label == "fire":
                    detect_fire = True
                elif label == "smoke":
                    detect_smoke = True

    # Logging
    if detect_fire and detect_smoke:
        logger.info("Fire and Smoke detected")
    elif detect_fire:
        logger.info("Fire detected")
    elif detect_smoke:
        logger.info("Smoke detected")
    else:
        logger.info("No Fire/Smoke detected")

    return {
        "flags": {
            "fire": detect_fire,
            "smoke": detect_smoke
        },
        "detections": detections
    }


# HSV FIRE DETECTION
def fire_detection_hsv(
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    thresh: int
) -> Dict[str, Any]:
    """
    Fire Detection using image processing technique. HSV and motion detection

    Args:
        - frame (np.ndarray): Image / Video Frame to be inferred
        - prev_frame (np.ndarray | None): Image / Video Frame used for motion.
        Previous frame to be passed here.
        - thresh (float): Threshold confidence for person detection

    Returns:
        - returns Dict[str, Any] which is dictionary of fire detection data.
    """

    fire_detected = False
    fire_boxes: List[Tuple[int, int, int, int]] = []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 80, 150])
    upper = np.array([35, 255, 255])

    color_mask = cv2.inRange(hsv, lower, upper)

    # Noise reduction
    color_mask = cv2.GaussianBlur(color_mask, (5, 5), 0)
    color_mask = cv2.morphologyEx(
        color_mask, cv2.MORPH_OPEN, np.ones((3, 3))
    )

    # Motion detection
    if prev_frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, gray)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        motion_mask = cv2.GaussianBlur(motion_mask, (5, 5), 0)
        motion_mask = cv2.morphologyEx(
            motion_mask, cv2.MORPH_OPEN, np.ones((3, 3))
        )

        mask = cv2.bitwise_and(color_mask, motion_mask)
    else:
        mask = color_mask

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > thresh:
            fire_detected = True

            x, y, w, h = cv2.boundingRect(contour)
            fire_boxes.append((x, y, x + w, y + h))

    if not fire_detected:
        logger.info("No fire detected (HSV fallback)")

    return {
        "flags": {"fire": fire_detected},
        "detections": [
            {
                "type": "fire",
                "bbox": box,
                "conf": 1.0
            }
            for box in fire_boxes
        ]
    }
