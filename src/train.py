import torch
from ultralytics import YOLO


def train_model():
    assert torch.cuda.is_available(), 'CUDA is not available'

    model = YOLO('yolo11n-seg.pt')
    model.train(
        data='chickens_ready_for_harvest.yaml',
        epochs=50,
        device=0,
    )
