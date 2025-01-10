import torch
from ultralytics import YOLO

from src.dataset import Dataset


def train_model(dataset: Dataset):
    assert torch.cuda.is_available(), 'CUDA is not available'

    model = YOLO('yolo11n-seg.pt')
    model.train(
        data=str(dataset.ultralytics_train_config.absolute()),
        epochs=50,
        device=0,
    )
