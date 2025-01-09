from pathlib import Path


path = Path('datasets/main/raw')

for file in path.iterdir():
    if 'img' in file.stem:
        num = file.stem.split('_')[1]
        num = int(num)

        new_file = f'sample_{num:04d}{file.suffix}'

        print(new_file)

        file.rename(file.with_name(new_file))
