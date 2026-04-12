from typing import Any, Dict, List, Optional

from loguru import logger

from utils.awareness import (
    calculate_hazard_zone,
    compute_distance,
    group_detections,
    is_inside_zone,
)


def mission_manager(
    detections: List[Dict],
    flags: Dict[str, bool],
    frame_shape
) -> Optional[Dict[str, Any]]:
    """
    Scoring-based mission manager.

    Converts perception outputs into a risk score and maps it to severity.

    Args:
        - detections: Unified detections (person, fire, smoke)
        - flags: Detection flags
        - frame_shape: Original frame shape

    Returns:
        - Decision dictionary
    """

    if not detections:
        logger.info("Nothing detected in frame")
        return None

    grouped = group_detections(detections)

    persons = grouped["person"]
    fires = grouped["fire"]
    smokes = grouped["smoke"]

    has_person = flags.get("person", False)
    has_fire = flags.get("fire", False)
    has_smoke = flags.get("smoke", False)

    score = 0
    details = []

    # BASE SIGNALS
    if has_fire:
        score += 3

    if has_smoke:
        score += 2

    if has_person:
        score += 1

    # SPATIAL REASONING
    for person in persons:
        person_box = person["bbox"]
        person_id = person.get("id", -1)

        # 🔥 FIRE INTERACTION
        for fire in fires:
            fire_box = fire["bbox"]
            fire_zone = calculate_hazard_zone(fire_box, frame_shape)

            if is_inside_zone(person_box, fire_zone):
                score += 5
                details.append({
                    "person_id": person_id,
                    "event": "inside_fire_zone"
                })

            else:
                distance = compute_distance(person_box, fire_box, frame_shape)
                if distance < 0.2:
                    score += 3
                    details.append({
                        "person_id": person_id,
                        "event": "near_fire",
                        "distance": distance
                    })

        # 🌫️ SMOKE INTERACTION
        for smoke in smokes:
            smoke_box = smoke["bbox"]
            smoke_zone = calculate_hazard_zone(smoke_box, frame_shape)

            if is_inside_zone(person_box, smoke_zone):
                score += 3
                details.append({
                    "person_id": person_id,
                    "event": "inside_smoke_zone"
                })

            else:
                distance = compute_distance(person_box, smoke_box, frame_shape)
                if distance < 0.25:
                    score += 2
                    details.append({
                        "person_id": person_id,
                        "event": "near_smoke",
                        "distance": distance
                    })

    # SCORE → SEVERITY MAPPING
    if score >= 8:
        severity = "CRITICAL"
        action = "IMMEDIATE_EVACUATION"
        alert = "EXTREME_SAFETY_RISK"

    elif score >= 5:
        severity = "HIGH"
        action = "ALERT_AND_MONITOR"
        alert = "HIGH_RISK_DETECTED"

    elif score >= 3:
        severity = "MEDIUM"
        action = "MONITOR_AND_ALERT"
        alert = "MODERATE_RISK"

    else:
        severity = "LOW"
        action = "LOG_AND_MONITOR"
        alert = "LOW_RISK"

    # Special handling for only person (no hazard context)
    if has_person and not has_fire and not has_smoke:
        return {
            "alert": "HUMAN_PRESENCE_DETECTED",
            "severity": "LOW",
            "action": "LOG_AND_MONITOR",
            "context": "UNKNOWN_ENVIRONMENT",
            "score": score,
            "details": details
        }

    return {
        "alert": alert,
        "severity": severity,
        "action": action,
        "score": score,
        "details": details
    }
