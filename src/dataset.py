from pathlib import Path

from dataclasses import dataclass


def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_dir(path: Path):
    if not path.exists():
        path.mkdir()
        return

    for item in path.rglob('*'):
        item.unlink()


@dataclass
class Dataset:
    name: str

    @property
    def root_dir(self):
        return Path(f'datasets/{self.name}')

    @property
    def raw_dir(self):
        return self.root_dir / 'raw'

    @property
    def augmented_dir(self):
        return self.root_dir / 'augmented'

    @property
    def coco_dir(self):
        return self.root_dir / 'coco'

    @property
    def train_dir(self):
        return self.root_dir / 'train'

    @property
    def val_dir(self):
        return self.root_dir / 'val'

    def create_augmented_dir(self):
        mkdir(self.augmented_dir)

    def create_coco_dir(self):
        mkdir(self.coco_dir)

    def create_train_dir(self):
        mkdir(self.train_dir)

    def create_val_dir(self):
        mkdir(self.val_dir)

    def clean_augmented_dir(self):
        clean_dir(self.augmented_dir)

    def clean_coco_dir(self):
        clean_dir(self.coco_dir)

    def clean_train_dir(self):
        clean_dir(self.train_dir)

    def clean_val_dir(self):
        clean_dir(self.val_dir)

    def iterate_raw_samples(self):
        for file in self.raw_dir.iterdir():
            if file.suffix == '.png':
                yield RawSample(self, file.stem)

    def iterate_augmentations(self):
        for file in self.augmented_dir.iterdir():
            if file.suffix == '.png' and not file.stem.endswith('_preview'):
                _, image_id, _, augmentation_id = file.stem.split('_')
                yield AugmentedSample(self, image_id, augmentation_id)


@dataclass
class AugmentedSample:
    dataset: Dataset
    image_id: str
    augmentation_id: str

    @property
    def image_path(self):
        return self.dataset.augmented_dir / f'{self.image_id}_{self.augmentation_id}.png'

    @property
    def annotations_path(self):
        return self.dataset.augmented_dir / f'{self.image_id}_{self.augmentation_id}.json'

    @property
    def preview_path(self):
        return self.dataset.augmented_dir / f'{self.image_id}_{self.augmentation_id}_preview.png'


@dataclass
class RawSample:
    dataset: Dataset
    name: str

    @property
    def image_path(self):
        return self.dataset.raw_dir / f'{self.name}.png'

    @property
    def annotations_path(self):
        return self.dataset.raw_dir / f'{self.name}.json'
