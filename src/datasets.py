import json
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

    def __post_init__(self):
        self.path = Path(self.path).resolve()
        self.raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        self.samples_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    @property
    def raw_dir(self):
        return self.path / 'raw'

    @property
    def samples_dir(self):
        return self.path / 'samples'

    def get_samples(self):
        items = self.samples_dir.iterdir()
        items = list(items)

        samples = []

        for item in items:
            sample = Sample(path=item)
            samples.append(sample)

        return samples

    def create_sample_dir(self):
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

    def destroy(self):
        for item in self.path.rglob('*'):
            item.unlink()
        self.path.rmdir()

    def copy_image_from(self, source: Path):
        source = Path(source).resolve()
        self.image_path.write_bytes(source.read_bytes())

    def write_polygons(self, data: dict):
        self.polygons_path.write_text(
            data=json.dumps(data, indent=4)
        )


def get(name: str) -> Dataset:
    return Dataset.create(path=Path('datasets') / name)
