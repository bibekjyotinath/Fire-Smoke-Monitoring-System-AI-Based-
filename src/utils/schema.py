from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    type: str
    bbox: Tuple[int, int, int, int]
    conf: float
    id: int = -1

    def validate(self):
        """
        Validate the schema of detection data.
        """
        if self.type not in ["person", "fire", "smoke"]:
            raise ValueError(f"Invalid type: {self.type}")

        if not isinstance(self.bbox, tuple) or len(self.bbox) != 4:
            raise ValueError(f"Invalid bbox: {self.bbox}")

        for v in self.bbox:
            if not isinstance(v, int):
                raise ValueError(f"Bbox must be int: {self.bbox}")
