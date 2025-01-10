"""
Aborting using onnx runtime since using Ultralytics to train the model, it's best
to use the same for inference.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

_context = {}


def get_value(key: str):
    return _context.get(key)


def set_value(key: str, value):
    _context[key] = value


@dataclass
class Prediction:
    masks: list[np.ndarray]
    classes: list[str]


def predict(image: np.ndarray, class_names: dict[int, str]):
    session = get_value('onnx_session')

    if not session:
        session = onnxruntime.InferenceSession('datasets/wheat_fields/model.onnx')
        set_value('onnx_session', session)

    input_name = session.get_inputs()[0].name
    input_dims = (640, 640)
    input_w, input_h = input_dims

    image_dims = image.shape[0:2][::-1]
    image_w, image_h = image_dims

    # resize image
    ratio = min(input_w / image_w, input_h / image_h)
    new_w = int(image_w * ratio)
    new_h = int(image_h * ratio)

    resized = cv2.resize(image, (new_w, new_h))

    # calculate padding
    pad_w = (input_w - new_w) // 2
    pad_h = (input_h - new_h) // 2

    # pad image
    padded = cv2.copyMakeBorder(
        resized,
        pad_h,
        pad_h,
        pad_w,
        pad_w,
        cv2.BORDER_CONSTANT,
        value=(115, 115, 115)
    )

    # convert to float32, scale, and NCHW
    processed = padded.astype(np.float32) / 255.0
    processed = np.transpose(processed, (2, 0, 1))
    processed = np.expand_dims(processed, axis=0)

    outputs = session.run(None, {input_name: processed})

    masks_n = 32

    x, protos = outputs[0], outputs[1]

    x = np.einsum('bcn->bnc', x)

    return outputs


def main():
    image = cv2.imread('test_images/test_wheat_fields/img_1.png')
    classes = {}

    results = predict(image, classes)

    print(results)


if __name__ == '__main__':
    # main()
    pass
