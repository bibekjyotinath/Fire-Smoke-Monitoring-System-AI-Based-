import argparse
import json
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import cv2
import torch
from loguru import logger
from ultralytics.models import YOLO

from reader.reader_factory import ReaderFactory
from utils.awareness import get_zone, map_detections_to_original
from utils.detection import (
    fire_smoke_detection,
    person_detection,
)
from utils.image_process import resize_image
from utils.schema import Detection

# LOGGING
logger.add(
    "pipeline.log",
    rotation="10 MB",
    retention="10 days",
    level="DEBUG",
    format="{time} | {level} | {message}"
)


# VIDEO CONVERSION
def convert_to_h264(input_path: str, output_path: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-vcodec", "libopenh264",
                "-acodec", "aac",
                output_path
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg failed:\n{result.stderr}")
            return False

        return True

    except Exception as e:
        logger.error(f"FFmpeg exception: {e}")
        return False


# MISSION MANAGER LOADER
def get_mission_manager(mode: str):
    if mode == "rules":
        from manager.rule import mission_manager
    elif mode == "scoring":
        from manager.score import mission_manager
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return mission_manager


# MAIN PIPELINE
def main(args):

    output_dir = Path(args.output)

    images_dir = output_dir / "images"
    video_dir = output_dir / "videos"
    json_dir = output_dir / "json"

    images_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    event_log_path = json_dir / "events.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    person_model = YOLO(args.person_model).to(device)
    fire_model = YOLO(args.fire_model).to(device)

    mission_manager_fn = get_mission_manager(args.mode)

    if Path(args.source).is_dir():
        source = list(Path(args.source).glob("*"))
    else:
        source = args.source

    data_reader = ReaderFactory.get_reader(source=source)

    # VIDEO STATE
    video_writer = None
    recording = False
    record_frames_left = 0
    clip_length = 50
    video_path = None
    converted_video_path = None

    # SMOOTHING
    severity_history = deque(maxlen=5)

    prev_severity = None
    events_log: List[Dict[str, Any]] = []

    for idx, data in enumerate(data_reader.read_data()):

        original_frame = data["frame"]
        if original_frame is None:
            continue

        # PREPROCESS
        resized_frame, scale, pad = resize_image(
            img=original_frame,
            resize=args.size,
            with_pad=True
        )

        # DETECTION
        try:
            person_result = person_detection(resized_frame, person_model, args.person_conf)
            fire_result = fire_smoke_detection(resized_frame, fire_model, args.fire_conf)
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            continue

        detections = []
        detections += person_result.get("detections", [])
        detections += fire_result.get("detections", [])

        # FLAGS
        flags = {
            **person_result["flags"],
            **fire_result["flags"]
        }

        # VALIDATE
        validated_detections = []
        for d in detections:
            try:
                det = Detection(
                    type=d["type"],
                    bbox=d["bbox"],
                    conf=d.get("conf", 1.0),
                    id=d.get("id", -1)
                )
                det.validate()
                validated_detections.append(d)
            except Exception:
                continue

        # MAP
        mapped_detections = map_detections_to_original(
            validated_detections,
            scale,
            pad,
            original_frame.shape
        )

        # SCENE SUMMARY
        scene_summaries = []

        for d in mapped_detections:
            zone = get_zone(d["bbox"], original_frame.shape)
            summary = f"{d['type']} detected in {zone}"
            scene_summaries.append(summary)

        scene_summary = "; ".join(scene_summaries) \
        if scene_summaries else "No hazards detected"


        # MISSION
        try:
            mission_output = mission_manager_fn(
                detections=mapped_detections,
                flags=flags,
                frame_shape=original_frame.shape
            )
        except Exception:
            mission_output = None

        # SMOOTHING
        raw_severity = mission_output.get("severity", "") if mission_output else ""
        severity_history.append(raw_severity)

        severity = raw_severity
        if severity_history.count("CRITICAL") >= 3:
            severity = "CRITICAL"
        elif severity_history.count("HIGH") >= 3:
            severity = "HIGH"

        # VISUALIZE
        annotated_frame = original_frame.copy()

        for d in mapped_detections:
            x1, y1, x2, y2 = d["bbox"]

            if d["type"] == "person":
                color = (0, 255, 0)        # green
            elif d["type"] == "fire":
                color = (0, 0, 255)        # red
            elif d["type"] == "smoke":
                color = (0, 165, 255)      # orange
            else:
                color = (255, 255, 255)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        combined_frame = cv2.hconcat([original_frame, annotated_frame])

        # SAVE IMAGE
        image_path = None
        if severity in ["HIGH", "CRITICAL"]:
            timestamp = int(time.time() * 1000)
            image_path = images_dir / f"{severity}_Frame_{idx}_{timestamp}.jpg"

            try:
                cv2.imwrite(str(image_path), annotated_frame)
            except Exception as e:
                logger.error(f"Image save failed: {e}")

        # VIDEO TRIGGER
        trigger_event = severity in ["HIGH", "CRITICAL"] and prev_severity not in ["HIGH", "CRITICAL"]

        if trigger_event and not recording:
            recording = True
            record_frames_left = clip_length

            timestamp = int(time.time() * 1000)

            video_path = video_dir / f"{severity}_Clip_{idx}_{timestamp}.mp4"
            converted_video_path = Path(str(video_path).replace(".mp4", "_h264.mp4"))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                20,
                (combined_frame.shape[1], combined_frame.shape[0])
            )

        # VIDEO WRITE
        try:
            if recording and video_writer is not None:
                video_writer.write(combined_frame)
                record_frames_left -= 1

                if record_frames_left <= 0:
                    recording = False

                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None

                    success = convert_to_h264(
                        str(video_path), str(converted_video_path)
                    )

                    if success:
                        logger.info(
                            f"Converted video saved: {converted_video_path}"
                        )

                        if video_path is not None and video_path.exists():
                            video_path.unlink()
                    else:
                        logger.error(
                            "Video conversion failed. Keeping original file."
                        )

        except Exception as e:
            logger.error(f"Video write failed: {e}")
            recording = False

        # MISSION CONFIDENCE
        weights = {
            "fire": 3,
            "smoke": 2,
            "person": 1
        }

        weighted_sum = sum(d["conf"] * weights.get(d["type"], 1) for d in mapped_detections)
        total_weight = sum(weights.get(d["type"], 1) for d in mapped_detections)

        if total_weight == 0:
            mission_confidence = 0.0
        else:
            mission_confidence = weighted_sum / total_weight

        # EVENT LOG
        if mission_output:
            event_id = f"{severity}_{idx}_{int(time.time() * 1000)}"

            events_log.append({
                "event_id": event_id,
                "frame_id": idx,
                "timestamp": time.time(),
                "severity": severity,
                "scene_summary": scene_summary,
                "zones": list(set([get_zone(d["bbox"], original_frame.shape) for d in mapped_detections])),
                "confidence": mission_confidence,
                "alert": mission_output.get("alert"),
                "action": mission_output.get("action"),
                "details": mission_output.get("details", []),
                "image_path": str(image_path) if image_path else None,
                "video_path": str(converted_video_path) if converted_video_path else None
            })

        prev_severity = severity

    # SAVE JSON
    try:
        with open(event_log_path, "w") as f:
            json.dump(events_log, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save events.json: {e}")



# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str,
        default=str(Path(__file__).parent.parent / "Inference"))

    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--person_model", type=str, default="yolov8m.pt")
    parser.add_argument("--fire_model", type=str, default="fire_smoke.pt")

    parser.add_argument("--person_conf", type=float, default=0.6)
    parser.add_argument("--fire_conf", type=float, default=0.6)
    parser.add_argument("--fire_area_thresh", type=int, default=50)

    parser.add_argument("--mode", type=str, default="scoring",
        choices=["rules", "scoring"])

    args = parser.parse_args()
    main(args)
