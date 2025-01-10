import json
from dataclasses import dataclass
from pathlib import Path


def mkdir(path: Path):
    path.mkdir(
        parents=True,
        exist_ok=True
    )


@dataclass
class Dataset:
    path: Path

    def __post_init__(self):
        self.path = Path(self.path).resolve()

    def create_dirs(self):
        mkdir(self.path)
        mkdir(self.raw_dir)
        mkdir(self.coco_dir)
        mkdir(self.train_dir)

    @property
    def raw_dir(self):
        return self.path / 'raw'

    @property
    def coco_dir(self):
        return self.path / 'coco'

    @property
    def train_dir(self):
        return self.path / 'train'

    @property
    def val_dir(self):
        return self.path / 'val'

    @property
    def ultralytics_config(self):
        return self.path / 'ultralytics.yaml'

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