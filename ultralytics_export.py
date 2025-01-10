import torch.jit
from ultralytics import YOLO


def export_model():
    model = YOLO('datasets/wheat_fields/model.pt')
    model.export(format='torchscript', imgsz=640, simplify=True)
