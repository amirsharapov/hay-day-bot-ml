import torch
from ultralytics import YOLO

from src.dataset import Dataset


def train_model(dataset: Dataset, model: str = 'yolo11s-seg.pt'):
    model = YOLO(model)
    model.train(
        data=str(dataset.ultralytics_train_config.absolute()),
        epochs=50,
        device=0 if torch.cuda.is_available() else 'cpu'
    )
