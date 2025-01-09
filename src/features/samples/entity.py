from dataclasses import dataclass
from pathlib import Path

import cv2

from src.lib.metadata_file import MetadataFile


@dataclass
class Sample:
    path: Path

    @property
    def image_path(self):
        return self.path / 'image.jpg'

    @property
    def polygons_path(self):
        return self.path / 'polygons.json'

    def read_image(self):
        return cv2.imread(str(self.image_path))

    def read_polygons(self):
        polygons = []
        file = MetadataFile.from_path(self.polygons_path)

        for polygon in file.get_key('polygons'):
            pass


@dataclass
class Polygon:
    label: str
    points: list[tuple[int, int]]
