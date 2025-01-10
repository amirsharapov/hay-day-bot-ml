import cv2
import onnx
import onnxruntime as rt
import torch.jit
from ultralytics import YOLO


def export_model():
    model = YOLO('datasets/wheat_fields/model.pt')
    model.export(format='onnx', imgsz=640, dynamic=True)


def test_onnx():
    model = onnx.load('datasets/wheat_fields/model.onnx')
    onnx.checker.check_model(model)


def test_onnx_inference():
    session = rt.InferenceSession('datasets/wheat_fields/model.onnx')

    inputs = session.get_inputs()

    image = cv2.imread('test_images/test_wheat_fields/img_1.png')
    image = cv2.resize(image, (640, 640))

    image = image[:, :, ::-1]
    image = image.transpose(2, 0, 1)

    image = image.astype('float32') / 255.0
    image = image[None, :, :, :]

    outputs = session.run(None, {'images': image})

    print(outputs)


if __name__ == '__main__':
    # test_onnx()
    test_onnx_inference()
    # export_model()
