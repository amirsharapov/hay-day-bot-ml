import json
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path

import cv2

from imgaug.augmentables.polys import Polygon
from imgaug import augmenters as iaa, PolygonsOnImage
from ultralytics import YOLO

from src.dataset import Dataset


# Copied from https://imgaug.readthedocs.io/en/latest/source/examples_basics.html
SEQUENTIAL = iaa.Sequential(
    children=[
        iaa.Sometimes(
            0.5,
            iaa.Fliplr()
        ),
        iaa.Sometimes(
            0.5,
            iaa.Crop(percent=(0, 0.1))
        ),
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.3))
        ),
        iaa.Sometimes(
            0.5,
            iaa.LinearContrast((0.95, 1.05))
        ),
        iaa.Sometimes(
            0.9,
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03 * 255), per_channel=0.3)
        ),
        iaa.Sometimes(
            0.9,
            iaa.Multiply(mul=(0.8, 1.2), per_channel=0.2)
        ),
        iaa.Sometimes(
            0.9,
            iaa.Affine(rotate=0.0, shear=(-8, 8))
        ),
        iaa.Sometimes(
            0.5,
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.8, 1.2))
        ),
        iaa.Sometimes(
            0.2,
            iaa.Grayscale(alpha=(0.0, 0.2)),
        )
    ],
    random_order=True
)


def generate_augmentations(dataset: Dataset, from_scratch: bool = False):
    if from_scratch:
        dataset.clean_augmented_dir()

    for sample in dataset.iterate_raw_samples():
        # Skip if the sample already has augmentations
        if not from_scratch:
            pass

        # An array of all the images to save
        images_to_save = []
        images_polygons_to_save = []

        image = cv2.imread(str(sample.image_path))

        data = json.loads(sample.annotations_path.read_text())

        polygons = []

        for shape in data['shapes']:
            points = [(int(x), int(y)) for x, y in shape['points']]
            polygons.append(Polygon(points))

        polygons = PolygonsOnImage(
            polygons,
            shape=image.shape
        )

        # Keep the original image and polygons
        images_to_save.append(image)
        images_polygons_to_save.append(polygons)

        # Create N augmented versions of the image
        n = 32

        image_copies = [deepcopy(image) for _ in range(n)]
        image_polygons_copies = [deepcopy(polygons) for _ in range(n)]

        # Augment the images
        for image_copy, image_polygon_copy in zip(image_copies, image_polygons_copies):
            aug_image, aug_polygons = SEQUENTIAL(
                image=image_copy,
                polygons=image_polygon_copy
            )

            images_to_save.append(aug_image)
            images_polygons_to_save.append(aug_polygons)

        # Save the images and annotations
        for i, (image_to_save, image_polygons_to_save) in enumerate(zip(images_to_save, images_polygons_to_save)):
            name = f'{sample.name}_aug_{i}'

            data = {
                'shapes': []
            }

            for polygon in image_polygons_to_save.clip_out_of_image().items:
                polygon: Polygon
                data['shapes'].append({
                    'label': 'Chicken',
                    'points': [[int(x), int(y)] for x, y in polygon.exterior]
                })

            image_path = dataset.augmented_dir / f'{name}.png'
            cv2.imwrite(str(image_path), image_to_save)

            annotations_path = dataset.augmented_dir / f'{name}.json'
            annotations_path.write_text(json.dumps(data, indent=4))

            # Generate a preview image
            preview_image = cv2.imread(str(image_path))

            for polygon in data['shapes']:
                points = polygon['points']
                points = [(x, y) for x, y in points]

                for j in range(len(points)):
                    cv2.line(preview_image, points[j], points[(j + 1) % len(points)], (0, 255, 0), 2)

            preview_path = dataset.augmented_dir / f'{name}_preview.png'
            cv2.imwrite(str(preview_path), preview_image)


def generate_coco_dir_from_augmented_dir(dataset: Dataset, from_scratch: bool = False):
    if from_scratch:
        dataset.clean_coco_dir()

    files = [file for file in dataset.augmented_dir.glob('*.png') if not file.stem.endswith('_preview')]
    files = list(sorted(files, key=lambda _: _.stem))

    dims = None
    classes = {}

    for i, file in enumerate(files):
        if file.stem.endswith('_preview'):
            continue

        sample_name = 'sample_{:04d}'.format(i)

        image = cv2.imread(str(file))

        if dims is None:
            dims = image.shape[:2]

        else:
            assert dims == image.shape[:2]

        cv2.imwrite(str(dataset.coco_dir / f'{sample_name}.png'), image)

        rows = []
        data = json.loads(file.with_suffix('.json').read_text())

        for shape in data['shapes']:
            label = shape['label']
            classes[label] = classes.get(label, len(classes))

            class_id = classes[label]

            row = f'{class_id} '

            for point in shape['points']:
                x_normalized = point[0] / dims[1]
                y_normalized = point[1] / dims[0]

                row += f'{x_normalized:.6f} {y_normalized:.6f} '

            rows.append(row.strip())

        with open(dataset.coco_dir / f'{sample_name}.txt', 'w') as f:
            f.write('\n'.join(rows))


def preview_augmented_annotations(dataset: Dataset):
    files = dataset.augmented_dir.glob('*_preview.png')
    files = list(sorted(files, key=lambda _: _.stem))

    for file in files:
        file = str(file)
        image = cv2.imread(file)

        cv2.imshow(file, image)
        cv2.waitKey(0)

        cv2.destroyWindow(file)


def generate_train_val_dir(dataset: Dataset):
    dataset.clean_train_dir()
    dataset.clean_val_dir()

    files = [file for file in dataset.coco_dir.glob('*.png')]
    files = list(sorted(files, key=lambda _: _.stem))

    # shuffle files
    random.shuffle(files)

    train_files = files[:int(len(files) * 0.8)]
    val_files = files[int(len(files) * 0.8):]

    for file in train_files:
        file: Path

        annotations_file = file.with_suffix('.txt')

        target_image_file = dataset.train_dir / file.name
        target_annotations_file = dataset.train_dir / annotations_file.name

        shutil.copyfile(file, target_image_file)
        shutil.copyfile(annotations_file, target_annotations_file)

    for file in val_files:
        file: Path

        annotations_file = file.with_suffix('.txt')

        target_image_file = dataset.val_dir / file.name
        target_annotations_file = dataset.val_dir / annotations_file.name

        shutil.copyfile(file, target_image_file)
        shutil.copyfile(annotations_file, target_annotations_file)


def train_model():
    model = YOLO('yolo11n-seg.pt')
    model.train(
        data='train_chicken_detection_yolo_config.yaml',
        epochs=100,
        device=0
    )


def test_model():
    model = YOLO('runs/segment/train4/weights/best.pt')

    image = cv2.imread('img_1.png')
    model(image, device='cpu')

    start = time.time()
    results = model(image)
    elapsed = time.time() - start

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        x, y, w, h = int(x), int(y), int(w), int(h)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Elapsed:', elapsed)
    print(results)


def main():
    dataset = Dataset('main')

    generate_augmentations(dataset)
    # preview_augmented_annotations(dataset)
    generate_coco_dir_from_augmented_dir(dataset)
    generate_train_val_dir(dataset)
    # train_model()
    # test_model()


if __name__ == '__main__':
    main()
