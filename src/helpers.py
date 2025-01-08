from pathlib import Path


def mkdir(path: Path):
    if not path.exists():
        path.mkdir()
    return path
