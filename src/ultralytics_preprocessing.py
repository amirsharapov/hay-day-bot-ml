import math
import random

import cv2
import yaml

from src.dataset import Dataset


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


def split_coco_into_train_val(dataset: Dataset):
    files = dataset.coco_dir.glob('*.txt')
    files = list(files)

    random.shuffle(files)

    train_val_ratio = 0.8
    train_count = math.floor(len(files) * train_val_ratio)

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


def preprocess_dataset(dataset: Dataset):
    dataset.create_dirs()

    convert_raw_to_coco(dataset)
    split_coco_into_train_val(dataset)
    generate_ultralytics_config_yaml(dataset)
