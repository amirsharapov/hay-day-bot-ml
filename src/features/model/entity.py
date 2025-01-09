from dataclasses import dataclass
from pathlib import Path


_METADATA_FILENAME = 'metadata.json'
_MODEL_FILENAME = 'model.pt'

@dataclass
class Model:
    path: Path

    @property
    def model_path(self):
        return self.path / _MODEL_FILENAME

    @property
    def metadata_path(self):
        return self.path / _METADATA_FILENAME

    def set_metadata_key(self, key: str, value):
        pass
