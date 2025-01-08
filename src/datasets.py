from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path('datasets')

AUGMENTED_DIR = ROOT_DIR / 'augmented'
COCO_DIR = ROOT_DIR / 'coco'
RAW_DIR = ROOT_DIR / 'raw'
TRAIN_DIR = ROOT_DIR / 'train'
VAL_DIR = ROOT_DIR / 'val'


def mkdir(path: Path):
    if not path.exists():
        path.mkdir()
    return path


def remove_children_from_dir(path: Path):
    if not path.exists():
        path.mkdir()
        return

    for item in path.rglob('*'):
        item.unlink()


def clean_augmented_dir(dataset: str):
    remove_children_from_dir(AUGMENTED_DIR / dataset)


def clean_coco_dir(dataset: str):
    remove_children_from_dir(COCO_DIR / dataset)


def clean_train_dir(dataset: str):
    remove_children_from_dir(TRAIN_DIR / dataset)


def clean_val_dir(dataset: str):
    remove_children_from_dir(VAL_DIR / dataset)


@dataclass
class AugmentedSample:
    pass


@dataclass
class AugmentedSamples:
    path: Path = AUGMENTED_DIR

    def remove_all_children(self):
        remove_children_from_dir(self.path)


@dataclass
class RawSample:
    pass


@dataclass
class RawSamples:
    pass


augmented_samples = AugmentedSamples()
raw_samples = RawSamples()
