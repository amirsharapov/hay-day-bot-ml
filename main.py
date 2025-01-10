from pathlib import Path

from src.dataset import Dataset
from src.ultralytics_preprocessing import preprocess_dataset
from src.ultralytics_train import train_model


def main(name: str):
    dataset = Dataset(Path(f'datasets/{name}'))

    preprocess_dataset(dataset)
    train_model(dataset)


if __name__ == '__main__':
    main('chickens_ready_for_harvest')
