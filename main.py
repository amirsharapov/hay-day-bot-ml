import sys
from pathlib import Path

import torch.cuda

from src.dataset import Dataset
from src.ultralytics_preprocessing import (
    convert_raw_to_coco,
    generate_augmentations_from_coco,
    split_augmented_into_train_val,
    generate_ultralytics_config_yaml
)
from src.ultralytics_train import train_model


def main(name: str):
    dataset = Dataset(Path(f'datasets/{name}'))

    dataset.cleanup_dirs()
    dataset.create_dirs_if_not_exists()

    print('Converting raw to coco...')
    convert_raw_to_coco(dataset)

    print('Generating augmentations...')
    generate_augmentations_from_coco(dataset, augmentations_per_image=5)

    print('Splitting augmented into train and val...')
    split_augmented_into_train_val(dataset)

    print('Generating ultralytics config yaml...')
    generate_ultralytics_config_yaml(dataset)

    if torch.cuda.is_available():
        print('Training model...')
        train_model(dataset)

    else:
        print('CUDA is not available, skipping training...')

    # dataset.cleanup_dirs()


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    # dataset_name = 'chickens_ready_for_harvest'

    main(dataset_name)
