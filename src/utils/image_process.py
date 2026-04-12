from typing import Tuple

import cv2
import numpy as np


def resize_image(
    img: np.ndarray,
    resize: Tuple[int, int] | int,
    with_pad: bool
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resizing the image with padding or with out"""

    resize = (resize, resize) if isinstance(resize, int) else resize
    target_width, target_height = resize

    if not with_pad:
        return cv2.resize(
            img,
            resize,
            interpolation=cv2.INTER_LINEAR
        ), 1.0, (0, 0)
    else:
        org_height, org_width = img.shape[:2]

        # Calculating scaling factor
        scale = min(target_width / org_width, target_height / org_height)

        # Computing new dimensions
        new_width, new_height = int(org_width * scale), int(org_height * scale)

        # Resizing image
        resized_img = cv2.resize(
            img,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Calculating padding
        pad_width = target_width - new_width
        pad_height = target_height - new_height

        top, bottom = pad_height // 2, pad_height - (pad_height // 2)
        left, right = pad_width // 2, pad_width - (pad_width // 2)

        # Applying padding
        padded_img = cv2.copyMakeBorder(
            resized_img,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        return padded_img, scale, (left, top)
