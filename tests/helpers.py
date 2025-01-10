from pathlib import Path

from ultralytics import YOLO


def run_detection(model: YOLO, image_path: str | Path):
    result = model.predict(image_path)

    print(result)
