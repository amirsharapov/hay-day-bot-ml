import json
from pathlib import Path


def main(do_rename: bool = False):
    path = Path('datasets/chickens_ready_for_harvest/anylabeling')

    highest_sample_n = float('-inf')

    for file in path.rglob('sample_*.png'):
        if file.stem.startswith('sample_'):
            num = file.stem.split('_')[1]
            num = int(num)

            highest_sample_n = max(highest_sample_n, num)

    for png_file in path.rglob('*.png'):
        json_file = png_file.with_suffix('.json')

        if png_file.name.startswith('sample_'):
            continue

        new_stem = f'sample_{highest_sample_n:04d}'

        print(f'Renaming "{png_file.stem}.png" to "{new_stem}.png"')
        if do_rename:
            png_file.rename(png_file.with_stem(new_stem))

        if json_file.exists():
            print(f'Renaming "{json_file.stem}.json" to "{new_stem}.json"')

            if do_rename:
                json_file.rename(json_file.with_stem(new_stem))

                data = json_file.with_stem(new_stem).read_text()
                data = json.loads(data)

                data['imagePath'] = f'{new_stem}.png'

                json_file.with_stem(new_stem).write_text(json.dumps(data, indent=4))

        highest_sample_n += 1


if __name__ == '__main__':
    main(do_rename=False)
