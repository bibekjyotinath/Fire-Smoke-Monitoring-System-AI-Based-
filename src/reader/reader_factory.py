import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, List, Union

import cv2
from loguru import logger


class DataReader(ABC):
    def __init__(self, paths: Union[int, str, Path, List[Union[int, str, Path]]]) -> None:
        # Normalize input to a list
        raw_paths = paths if isinstance(paths, list) else [paths]

        self.paths = []
        for p in raw_paths:
            # 1. Handle Webcams (int or string digits like "0")
            if isinstance(p, int) or (isinstance(p, str) and p.isdigit()):
                self.paths.append(int(p))
            # 2. Handle Network Streams (RTSP, HTTP, etc.)
            elif isinstance(p, str) and "://" in p:
                self.paths.append(p)
            # 3. Handle Local Files
            else:
                self.paths.append(Path(p))

    @abstractmethod
    def read_data(self) -> Generator[Any, None, None]:
        ...


class VideoReader(DataReader):
    """Video reader class for Files, Streams, and Webcams"""

    def read_data(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream frames from video files, RTSP, or Webcams
        """
        frame_id = 0

        for source_id, source in enumerate(self.paths):
            # Validation for local files
            if isinstance(source, Path):
                if not source.exists():
                    raise FileNotFoundError(f"Video file not found: {source}")
                cap_source = str(source) # OpenCV requires string for file paths
            else:
                # Ints (webcams) and Strings (streams) pass directly to OpenCV
                cap_source = source

            cap = cv2.VideoCapture(cap_source)
            if not cap.isOpened():
                logger.error(f"Could not open source: {source}")
                continue

            logger.info(f"Successfully opened source: {source}")

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    yield {
                        "frame_id": frame_id,
                        "frame": frame,
                        "source_id": source_id,
                        "source": source,
                        "timestamp": time.time()
                    }
                    frame_id += 1
            finally:
                cap.release()


class ImageReader(DataReader):
    """Image reader class"""

    def read_data(self) -> Generator[Dict[str, Any], None, None]:
        """
        Read image(s) and yield (index, image)
        """
        for source_id, path in enumerate(self.paths):
            # Because of DataReader, we know `path` is already a Path object here
            if not isinstance(path, Path):
                raise TypeError(f"ImageReader expects file paths, got {type(path)}")

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            img = cv2.imread(str(path))

            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")

            yield {
                "frame": img,
                "source_id": source_id,
                "source": str(path),
                "timestamp": time.time()
            }


class ReaderFactory:
    """Factory design pattern for reader functionality"""

    @staticmethod
    def get_reader(source: Any) -> DataReader:
        # Get a sample to check (handles both lists and single inputs)
        sample = source[0] if isinstance(source, list) else source

        # 1. Handling Webcams (int or digit string)
        if isinstance(sample, int) or (isinstance(sample, str) and sample.isdigit()):
            return VideoReader(source)

        # 2. Handling Network Streams
        if isinstance(sample, str) and "://" in sample:
            return VideoReader(source)

        # 3. Handling Files
        sample_path = Path(sample)
        suffix = sample_path.suffix.lower()

        if suffix in [".mp4", ".avi", ".mov", ".mkv"]:
            return VideoReader(source)
        elif suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
            return ImageReader(source)
        else:
            raise ValueError(f"Unsupported File Type: {suffix}")
