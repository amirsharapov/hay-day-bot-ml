import json
from dataclasses import dataclass
from pathlib import Path


def mkdir(path: Path):
    path.mkdir(
        parents=True,
        exist_ok=True
    )


def rmdir(path: Path):
    for item in path.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()

    if path.exists():
        path.rmdir()


@dataclass
class Dataset:
    path: Path

    def __post_init__(self):
        self.path = Path(self.path).resolve()

    def create_dirs_if_not_exists(self):
        mkdir(self.path)
        mkdir(self.raw_dir)
        mkdir(self.coco_dir)
        mkdir(self.augmented_dir)
        mkdir(self.train_dir)
        mkdir(self.val_dir)

    def cleanup_dirs(self):
        self.create_dirs_if_not_exists()
        rmdir(self.coco_dir)
        rmdir(self.augmented_dir)
        rmdir(self.train_dir)
        rmdir(self.val_dir)

    @property
    def raw_dir(self):
        return self.path / 'raw'

    @property
    def coco_dir(self):
        return self.path / 'coco'

    @property
    def augmented_dir(self):
        return self.path / 'augmented'

    @property
    def train_dir(self):
        return self.path / 'train'

    @property
    def val_dir(self):
        return self.path / 'val'

    @property
    def ultralytics_train_config(self):
        return self.path / 'train.yaml'

    def iterate_raw_annotations(self):
        for item in self.raw_dir.glob('*.json'):
            labels = json.loads(item.read_text())
            labels = RawAnnotations(labels, self)

            yield labels


@dataclass
class RawAnnotations:
    data: dict
    dataset: Dataset

    @property
    def image_path(self):
        return self.dataset.raw_dir / self.data['imagePath']

    def iterate_shapes(self):
        for shape in self.data['shapes']:
            for point in shape['points']:
                point[0] = int(point[0])
                point[1] = int(point[1])

            yield {
                'label': shape['label'],
                'points': shape['points']
            }