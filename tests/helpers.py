from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class SegmentedInstance:
    name: str
    mask: np.ndarray
    original_image: np.ndarray


def predict(model_path: str | Path, image_path: str | Path):
    model_path = Path(model_path)
    image_path = Path(image_path)

    model = YOLO(model_path)

    results = model(image_path)
    results = results[0]

    final_results = []

    for i, result in enumerate(results):
        plot = result.plot()
        cv2.imwrite(f'out_{i}.png', plot)

        dims = result.orig_shape

        mask = result.masks.data[0].numpy()
        mask = np.where(mask > 0.5, 1, 0)
        mask = mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, (dims[1], dims[0]))

        class_id = result.boxes.cls[0].item()
        class_name = result.names[class_id]

        original_image = cv2.imread(str(image_path))
        original_image = cv2.resize(original_image, (dims[1], dims[0]))

        final_results.append(SegmentedInstance(
            name=class_name,
            mask=mask,
            original_image=original_image
        ))

    return final_results