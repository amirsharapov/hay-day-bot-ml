import cv2

from tests.helpers import predict


def test_wheat_fields():
    objects = predict(
        'datasets/wheat_fields/model.pt',
        'test_images/test_wheat_fields/img_3.png'
    )

    for object_ in objects:
        cv2.imshow(object_.name, object_.mask)
        cv2.imshow(object_.name + ' preview', object_.original_image)
        cv2.waitKey(0)
