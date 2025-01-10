import math
import random
from collections import defaultdict

import cv2
import numpy as np
import yaml
import albumentations as A

from src.dataset import Dataset


def convert_coco_x_y_to_polygon(line: str, image_x: int, image_y: int):
    line = line.split(' ')
    class_id = int(line[0])

    points = []
    for i in range(1, len(line), 2):
        x = float(line[i]) * image_x
        y = float(line[i + 1]) * image_y

        points.append((x, y))

    return {
        'class_id': class_id,
        'points': points,
    }


def convert_polygon_to_coco_x_y(polygon: dict, image_x: int, image_y: int):
    class_id = polygon['class_id']
    points = polygon['points']

    line = f'{class_id} '

    for x, y in points:
        x /= image_x
        y /= image_y
        line += f'{x} {y} '

    line = line.strip()

    return line


def get_classes(dataset: Dataset):
    classes = set()

    for annotations in dataset.iterate_raw_annotations():
        for shape in annotations.iterate_shapes():
            classes.add(shape['label'])

    classes = list(classes)
    classes.sort()

    return {_: i for i, _ in enumerate(classes)}


def flip_dict(d):
    return {v: k for k, v in d.items()}


def convert_raw_to_coco(dataset: Dataset):
    classes = get_classes(dataset)

    for annotations in dataset.iterate_raw_annotations():

        image = cv2.imread(str(annotations.image_path))

        image_x = image.shape[1]
        image_y = image.shape[0]

        txt_contents = ''

        for shape in annotations.iterate_shapes():
            txt_contents += f'{classes[shape["label"]]} '

            points = shape['points']

            for point in points:
                x = point[0] / image_x
                y = point[1] / image_y

                txt_contents += f'{x} {y} '

            txt_contents = txt_contents.strip()
            txt_contents += '\n'

        target_image_path = dataset.coco_dir / annotations.image_path.name
        target_annotations_path = dataset.coco_dir / annotations.image_path.with_suffix('.txt').name

        target_image_path.write_bytes(annotations.image_path.read_bytes())
        target_annotations_path.write_text(txt_contents)


def split_augmented_into_train_val(dataset: Dataset):
    files = dataset.augmented_dir.glob('*.txt')
    files = list(files)

    random.shuffle(files)

    train_val_ratio = 0.8
    train_count = math.floor(len(files) * train_val_ratio)

    print(f'Train count: {train_count}')

    for i, file in enumerate(files):
        if i < train_count:
            target_dir = dataset.train_dir
        else:
            target_dir = dataset.val_dir

        # Write annotations
        target_file = target_dir / file.name
        target_file.write_text(file.read_text())

        # Write image
        target_file = target_dir / file.with_suffix('.png').name
        target_file.write_bytes(file.with_suffix('.png').read_bytes())


def generate_ultralytics_config_yaml(dataset: Dataset):
    data = {
        'path': dataset.path.absolute().as_posix(),
        'train': 'train',
        'val': 'val',
        'names': flip_dict(get_classes(dataset))
    }

    contents = yaml.dump(data, sort_keys=False)
    dataset.ultralytics_config.write_text(contents)


def generate_augmentations_from_coco(dataset: Dataset):
    transform = A.Compose(
        transforms=[
            A.HorizontalFlip(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            # A.AutoContrast(p=0.2),
            A.GaussNoise(p=0.2),
            # A.RandomBrightnessContrast(brightness_limit=(0.8, 1.2), contrast_limit=(0.8, 1.2), p=0.2),
            A.Affine(translate_percent=(-0.1, 0.1), scale=(0.8, 1.2), p=0.2),
            A.Sharpen(alpha=(0, 1.0), lightness=(0.8, 1.2), p=0.2),
            A.ToGray(p=0.2),
        ]
    )

    for image_path in dataset.coco_dir.glob('*.png'):
        image = cv2.imread(str(image_path))

        image_x = image.shape[1]
        image_y = image.shape[0]

        annotations = image_path.with_suffix('.txt').read_text()
        annotations = [convert_coco_x_y_to_polygon(line, image_x, image_y) for line in annotations.split('\n') if line]

        mask_indices_by_class_id = defaultdict(list)
        masks = []

        for i, annotation in enumerate(annotations):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

            class_id = annotation['class_id']
            points = annotation['points']
            points = np.array(points, dtype=np.int32)

            # noinspection PyTypeChecker
            cv2.fillPoly(mask, [points], 255)

            masks.append(mask)
            mask_indices_by_class_id[class_id].append(i)

        for i in range(8):
            augmented = transform(image=image, masks=masks)

            augmented_image = augmented['image']
            augmented_masks = augmented['masks']

            augmented_annotations = []

            for class_id, indices in mask_indices_by_class_id.items():
                for index in indices:
                    mask = augmented_masks[index]

                    # convert to polygon
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = [contour.flatten().tolist() for contour in contours]

                    if len(contours) != 1:
                        continue

                    contour = contours[0]
                    points = []

                    for j in range(0, len(contour), 2):
                        x = contour[j]
                        y = contour[j + 1]

                        points.append((x, y))

                    augmented_annotations.append({
                        'class_id': class_id,
                        'points': points
                    })

            contents = ''

            for annotation in augmented_annotations:
                contents += convert_polygon_to_coco_x_y(annotation, image_x, image_y) + '\n'

            target_image_path = dataset.augmented_dir / (image_path.stem + f'_{i}.png')
            target_image_path.write_bytes(cv2.imencode('.png', augmented_image)[1].tobytes())

            target_annotations_path = dataset.augmented_dir / (image_path.stem + f'_{i}.txt')
            target_annotations_path.write_text(contents)



def preprocess_dataset(dataset: Dataset):
    dataset.remove_dirs_except_raw()
    dataset.create_dirs()

    convert_raw_to_coco(dataset)
    generate_augmentations_from_coco(dataset)
    split_augmented_into_train_val(dataset)
    generate_ultralytics_config_yaml(dataset)
