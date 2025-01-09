from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dataset:
    path: Path

    @classmethod
    def create(cls, path: Path):
        path.mkdir(
            parents=True,
            exist_ok=True
        )
        return cls(path=path)

    @property
    def samples_dir(self):
        return self.path / 'anylabeling'

    @property
    def train_dir(self):
        return self.path / 'train'

    @property
    def val_dir(self):
        return self.path / 'val'

    def get_samples(self):
        items = self.samples_dir.iterdir()
        items = list(items)

        samples = []

        for item in items:
            sample = Sample(path=item)
            samples.append(sample)

        return samples

    def create_new_sample(self):
        items = self.samples_dir.iterdir()
        items = list(items)

        next_sample_dir = self.samples_dir / f'sample_{len(items) + 1}'

        return Sample.create(
            path=next_sample_dir
        )


@dataclass
class Sample:
    path: Path

    @classmethod
    def create(cls, path: Path):
        path.mkdir(
            parents=True,
            exist_ok=True
        )
        return cls(path=path)

    @property
    def image_path(self):
        return self.path / 'image.png'

    @property
    def metadata_path(self):
        return self.path / 'metadata.json'

    @property
    def polygons_path(self):
        return self.path / 'polygons.json'


def get(name: str) -> Dataset:
    return Dataset.create(path=Path('datasets') / name)
