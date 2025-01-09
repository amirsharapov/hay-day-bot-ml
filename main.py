import sys

from src.dataset import Dataset
from src.dataset_augmentation import (
    generate_train_val_dir,
    generate_coco_dir_from_augmented_dir,
    generate_augmentations
)
from src.train import train_model


def main():
    args = sys.argv[1:]

    actions = []

    if '--generate-augmentations' in args:
        actions.append('generate-augmentations')

    if '--train-model' in args:
        actions.append('train-model')

    dataset = Dataset('main')
    dataset_augmentations_exists = (
        dataset.augmented_dir.exists() and
        list(dataset.augmented_dir.glob('*'))
    )

    if not dataset_augmentations_exists and 'generate-augmentations' not in actions:
        print('Augmented directory does not exist. Generating augmentations...')
        actions.insert(0, 'generate-augmentations')

    visited = set()

    for action in actions:
        if action in visited:
            print(f'Action "{action}" already executed. Skipping...')
            continue

        if action == 'generate-augmentations':
            generate_augmentations(dataset)
            generate_coco_dir_from_augmented_dir(dataset)
            generate_train_val_dir(dataset)

        if action == 'train-model':
            train_model()

        visited.add(action)


if __name__ == '__main__':
    main()
