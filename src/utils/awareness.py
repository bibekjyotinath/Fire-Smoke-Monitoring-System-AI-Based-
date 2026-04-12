import math
from typing import Any, Dict, List, Tuple


# CORE GEOMETRY FUNCTIONS
def map_box_to_original(
    box: Tuple[int, int, int, int],
    scale: float,
    pad: Tuple[int, int],
    frame_shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Map the bbox to the original image

    Args:
        - box (Tuple[int, int, int, int]): Bounding box data from detection.
        - scale (float): Scaling information based calculated after resizing
        the image.
        - pad (Tuple[int, int]): Padded information of height and width after
        resizing the image.
        - frame_shape (Tuple[int, int, int]): Original image shape.

    Returns:
        - returns the bounding box updated shape to be added on original frame
        as Tuple[int, int, int, int].
    """

    x1, y1, x2, y2 = box
    pad_x, pad_y = pad
    height, width = frame_shape[:2]

    x1 = int((x1 - pad_x) / scale)
    y1 = int((y1 - pad_y) / scale)
    x2 = int((x2 - pad_x) / scale)
    y2 = int((y2 - pad_y) / scale)

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    return (x1, y1, x2, y2)


# CALCULATE BOX CENTER
def box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate the center point of the bounding box.

    Args:
        - box (Tuple[int, int, int, int]): Bounding box data from detection.

    Returns:
        - returns the center position of the bounding box as Tuple[int, int].
    """
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


# COMPUTE DISTANCE
def compute_distance(
    person_bbox: Tuple[int, int, int, int],
    fire_bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
) -> float:
    """
    Normalized distance between person and fire

    Args:
        - person_bbox (Tuple[int, int, int, int]): Bounding box of person after
        detection.
        - fire_bbox (Tuple[int, int, int, int]): Bounding box of fire after
        detection.
        - frame_shape (Tuple[int, int, int]): Original image shape.

    Returns:
        - returns the distance computed between person and the fire based on
        the center point of each bounding box as float
    """

    px, py = box_center(person_bbox)
    fx, fy = box_center(fire_bbox)

    height, width = frame_shape[:2]

    distance = math.sqrt((px - fx) ** 2 + (py - fy) ** 2)
    diagonal = math.sqrt(width ** 2 + height ** 2)

    return distance / diagonal


# CALCULATE HAZARD ZONE
def calculate_hazard_zone(
    fire_box: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
    scale_factor: float = 0.2
) -> Tuple[int, int, int, int]:
    """
    Calculation of hazard zone based on the bounding box of fire detection.

    Args:
        - fire_box (Tuple[int, int, int, int]): Bounding box of fire after
        detection.
        - frame_shape (Tuple[int, int, int]): Original image shape.
        - scale_factor (float): Scaling factor to calculate the hazard zone
        bounding box area. Default is 0.2.

    Returns:
        - returns hazard zone bounding box as Tuple[int, int, int, int]
    """
    x1, y1, x2, y2 = fire_box
    height, width = frame_shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = int(box_width * scale_factor)
    pad_y = int(box_height * scale_factor)

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


# INSIDE HAZARD ZONE
def is_inside_zone(
    person_box: Tuple[int, int, int, int],
    zone_box: Tuple[int, int, int, int]
) -> bool:
    """
    Check if the person is inside the hazard zone.

    Args:
        - person_box (Tuple[int, int, int, int]): Updated bounding box region
        of the person detected.
        - zone_box (Tuple[int, int, int, int]): Hazard zone bounding box

    Returns:
        - returns True if person is inside the hazard zone else False as bool.
    """
    px, py = box_center(person_box)
    x1, y1, x2, y2 = zone_box

    return x1 <= px <= x2 and y1 <= py <= y2


# NORMALIZING BBOX
def _normalize_bbox(box: Any) -> Tuple[int, int, int, int]:
    """
    Normalize bbox to (x1, y1, x2, y2) format.

    Handles:
        - tuple/list: (x1, y1, x2, y2)
        - nested: [(x1, y1, x2, y2)]
        - numpy arrays / torch tensors

    Returns:
        Tuple[int, int, int, int]
    """

    # Handle tensor / numpy
    if hasattr(box, "tolist"):
        box = box.tolist()

    # Handle nested structure
    if isinstance(box, (list, tuple)) and len(box) == 1:
        box = box[0]

    # Final validation
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"Invalid bbox format: {box}")

    x1, y1, x2, y2 = box

    return (int(x1), int(y1), int(x2), int(y2))


# MAP DETECTION
def map_detections_to_original(
    detections: List[Dict],
    scale: float,
    pad: Tuple[int, int],
    frame_shape: Tuple[int, int, int]
) -> List[Dict]:
    """
    Map all detections to original frame.

    Args:
        - detections (List[Dict]): List of detection dictionaries containing
        bounding box information.
        - scale (float): Scaling factor used during preprocessing.
        - pad (Tuple[int, int]): Padding applied during preprocessing.
        - frame_shape (Tuple[int, int, int]): Original image shape.

    Returns:
        - returns List[Dict] with updated bounding boxes mapped to original
        image coordinates.
    """

    mapped: List[Dict] = []

    for d in detections:
        try:
            # 🔥 Normalize FIRST (critical fix)
            norm_box = _normalize_bbox(d["bbox"])

            # Map to original
            mapped_box = map_box_to_original(
                norm_box, scale, pad, frame_shape
            )

            new_d = d.copy()
            new_d["bbox"] = mapped_box
            mapped.append(new_d)

        except Exception as e:
            # 🚨 Fail-safe (prevents pipeline crash)
            print(f"[WARNING] Skipping invalid bbox: {d.get('bbox')} | Error: {e}")
            continue

    return mapped


# GROUP DETECTION BASED ON TYPE
def group_detections(
    detections: List[Dict]
) -> Dict[str, List[Dict]]:
    """
    Group detections based on their type.

    Args:
        - detections (List[Dict]): List of detection dictionaries.

    Returns:
        - returns Dict[str, List[Dict]] where detections are grouped into
        categories such as person, fire, and smoke.
    """

    grouped = {
        "person": [],
        "fire": [],
        "smoke": []
    }

    for d in detections:
        grouped[d["type"]].append(d)

    return grouped


# FILTER BASED ON DETECTION
def filter_detections_by_type(
    detections: List[Dict],
    dtype: str
) -> List[Dict]:
    """
    Filter detections based on a specific type.

    Args:
        - detections (List[Dict]): List of detection dictionaries.
        - dtype (str): Type of detection to filter (e.g., person, fire, smoke).

    Returns:
        - returns List[Dict] containing only detections of the specified type.
    """

    return [d for d in detections if d["type"] == dtype]


# RELATIONSHIP COMPUTATION
def analyze_risk(
    detections: List[Dict],
    frame_shape: Tuple[int, int, int]
) -> List[Dict]:
    """
    Analyze person-fire relationships and compute risk metrics.

    Args:
        - detections (List[Dict]): List of detection dictionaries containing
        person and fire detections.
        - frame_shape (Tuple[int, int, int]): Original image shape.

    Returns:
        - returns List[Dict] containing interaction details such as distance,
        hazard zone status, and associated bounding boxes.
    """

    results = []

    grouped = group_detections(detections)

    persons = grouped["person"]
    fires = grouped["fire"]

    for fire in fires:
        fire_box = fire["bbox"]
        zone = calculate_hazard_zone(fire_box, frame_shape)

        for person in persons:
            person_box = person["bbox"]

            inside = is_inside_zone(person_box, zone)
            distance = compute_distance(person_box, fire_box, frame_shape)

            results.append({
                "person_id": person.get("id", -1),
                "fire_box": fire_box,
                "person_box": person_box,
                "inside_hazard": inside,
                "distance": distance
            })

    return results
