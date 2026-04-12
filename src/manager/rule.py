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
    Converts perception outputs into mission-level decisions.

    NOTE:
        No static zones are used. Decisions rely only on dynamic hazards
        such as fire and smoke.

    Args:
        - detections (List[Dict]): Unified detection list (person, fire, smoke)
        - flags (Dict[str, bool]): Detection flags (person, fire, smoke)
        - frame_shape (Tuple[int, int, int]): Shape of original frame

    Returns:
        - Structured mission decision dictionary
    """

    # No detections
    if not detections:
        logger.info("Nothing detected in frame")
        return None

    # Group detections
    grouped = group_detections(detections)

    persons = grouped["person"]
    fires = grouped["fire"]
    smokes = grouped["smoke"]

    has_person = flags.get("person", False)
    has_fire = flags.get("fire", False)
    has_smoke = flags.get("smoke", False)

    # Severity ranking
    severity_rank = {
        "CRITICAL": 3,
        "HIGH": 2,
        "MEDIUM": 1,
        "LOW": 0
    }

    def update_response(response, new_data):
        """Keep only highest severity decision"""
        if response["severity"] is None or \
           severity_rank[new_data["severity"]] > severity_rank[response["severity"]]:
            response.update(new_data)

    # Default response
    response: Dict[str, Any] = {
        "alert": None,
        "severity": None,
        "action": None,
        "details": []
    }

    # ---------------------------------------------------
    # 🔥🔥🔥 CASE 0: FIRE + SMOKE + PERSON (HIGHEST PRIORITY)
    # ---------------------------------------------------
    if has_fire and has_smoke and has_person:

        for fire in fires:
            fire_box = fire["bbox"]
            fire_zone = calculate_hazard_zone(fire_box, frame_shape)

            for smoke in smokes:
                smoke_box = smoke["bbox"]
                smoke_zone = calculate_hazard_zone(smoke_box, frame_shape)

                for person in persons:
                    person_box = person["bbox"]
                    person_id = person.get("id", -1)

                    in_fire = is_inside_zone(person_box, fire_zone)
                    in_smoke = is_inside_zone(person_box, smoke_zone)

                    if in_fire or in_smoke:
                        logger.info(
                            f"Person {person_id} inside combined fire/smoke hazard"
                        )

                        return {
                            "alert": "EXTREME_SAFETY_RISK",
                            "severity": "CRITICAL",
                            "action": "IMMEDIATE_EVACUATION",
                            "details": [{
                                "person_id": person_id,
                                "status": "inside_combined_hazard",
                                "fire_bbox": fire_box,
                                "smoke_bbox": smoke_box
                            }]
                        }

    # ---------------------------------------------------
    # 🔴 CASE 1: FIRE + PERSON
    # ---------------------------------------------------
    if has_fire and has_person:

        for fire in fires:
            fire_box = fire["bbox"]
            hazard_zone = calculate_hazard_zone(fire_box, frame_shape)

            for person in persons:
                person_box = person["bbox"]
                person_id = person.get("id", -1)

                inside = is_inside_zone(person_box, hazard_zone)
                distance = compute_distance(person_box, fire_box, frame_shape)

                if inside:
                    logger.info(f"Person {person_id} inside fire hazard zone")

                    return {
                        "alert": "CRITICAL_SAFETY_RISK",
                        "severity": "CRITICAL",
                        "action": "STOP_MISSION_AND_EVACUATE",
                        "details": [{
                            "person_id": person_id,
                            "status": "inside_fire_zone",
                            "distance": distance,
                            "fire_bbox": fire_box
                        }]
                    }

                elif distance < 0.2:
                    logger.info(f"Person {person_id} near fire")

                    update_response(response, {
                        "alert": "PERSON_NEAR_FIRE",
                        "severity": "HIGH",
                        "action": "ALERT_AND_MONITOR"
                    })

                    response["details"].append({
                        "person_id": person_id,
                        "status": "near_fire",
                        "distance": distance,
                        "fire_bbox": fire_box
                    })

        update_response(response, {
            "alert": "FIRE_DETECTED",
            "severity": "HIGH",
            "action": "ALERT_OPERATOR"
        })

    # ---------------------------------------------------
    # 🟠 CASE 2: SMOKE + PERSON
    # ---------------------------------------------------
    if has_smoke and has_person:

        for smoke in smokes:
            smoke_box = smoke["bbox"]
            smoke_zone = calculate_hazard_zone(smoke_box, frame_shape)

            for person in persons:
                person_box = person["bbox"]
                person_id = person.get("id", -1)

                inside = is_inside_zone(person_box, smoke_zone)
                distance = compute_distance(person_box, smoke_box, frame_shape)

                if inside:
                    logger.info(f"Person {person_id} inside smoke zone")

                    update_response(response, {
                        "alert": "PERSON_IN_SMOKE",
                        "severity": "HIGH",
                        "action": "ALERT_AND_MONITOR"
                    })

                    response["details"].append({
                        "person_id": person_id,
                        "status": "inside_smoke_zone",
                        "distance": distance,
                        "smoke_bbox": smoke_box
                    })

                elif distance < 0.25:
                    logger.info(f"Person {person_id} near smoke")

                    update_response(response, {
                        "alert": "PERSON_NEAR_SMOKE",
                        "severity": "MEDIUM",
                        "action": "MONITOR_CLOSELY"
                    })

                    response["details"].append({
                        "person_id": person_id,
                        "status": "near_smoke",
                        "distance": distance,
                        "smoke_bbox": smoke_box
                    })

        if response["severity"] is None:
            update_response(response, {
                "alert": "SMOKE_WITH_PERSON",
                "severity": "MEDIUM",
                "action": "MONITOR_AND_ALERT"
            })

    # ---------------------------------------------------
    # 🔴 CASE 3: ONLY FIRE
    # ---------------------------------------------------
    if has_fire:
        logger.info("Only fire detected")

        update_response(response, {
            "alert": "FIRE_DETECTED",
            "severity": "HIGH",
            "action": "STOP_MISSION_AND_ALERT_OPERATOR"
        })

    # ---------------------------------------------------
    # 🟡 CASE 4: ONLY SMOKE
    # ---------------------------------------------------
    if has_smoke:
        logger.info("Only smoke detected")

        update_response(response, {
            "alert": "SMOKE_DETECTED",
            "severity": "MEDIUM",
            "action": "MONITOR_AND_ALERT"
        })

    # ---------------------------------------------------
    # 🟢 CASE 5: ONLY PERSON
    # ---------------------------------------------------
    if has_person and not has_fire and not has_smoke:
        logger.info("Only person detected (no hazard context)")

        update_response(response, {
            "alert": "HUMAN_PRESENCE_DETECTED",
            "severity": "LOW",
            "action": "LOG_AND_MONITOR",
            "context": "UNKNOWN_ENVIRONMENT"
        })

    return response
