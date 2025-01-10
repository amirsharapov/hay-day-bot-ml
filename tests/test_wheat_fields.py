from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def test_wheat_fields():
    model = YOLO('datasets/wheat_fields/model.pt')

    result = model('test_images/test_wheat_fields/img.png')
    canvas = result[0].plot()

    cv2.imwrite('out.png', canvas)

    return

    canvas = np.zeros((result[0].orig_shape[0], result[0].orig_shape[1], 3), dtype=np.uint8)

    for mask in result[0].masks.data:
        mask = mask.numpy()
        mask = np.where(mask > 0.5, 1, 0)
        mask = mask.astype(np.uint8) * 255

        canvas = cv2.bitwise_or(canvas, mask)

    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)

