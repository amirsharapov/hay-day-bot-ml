import torch.jit
from ultralytics import YOLO


def export_model():
    model = YOLO('yolo11n-seg.pt')

    model_scripted = torch.jit.script(model)
    model_scripted.save('model_scripted.pt')
