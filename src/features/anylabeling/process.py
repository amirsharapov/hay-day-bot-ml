import json
from pathlib import Path

import cv2

from src.features.samples.entity import Sample


def _get_highest_sample_id():
    samples = list(Path('data/anylabeling').glob('sample_*'))

    if not samples:
        return 0

    return max(int(_.name.split('_')[1]) for _ in samples)


def _process_polygons_from_version_0_4_8(data:dict):
    shapes = data['shapes']

    data = {
        'polygons': []
    }

    for shape in shapes:
        for point in shape['points']:
            point[0] = int(point[0])
            point[1] = int(point[1])

        data['polygons'].append({
            'label': shape['label'],
            'points': shape['points']
        })

    return data


def process_labeled_images():
    """
    Migrates all the labeled anylabeling from the "anylabeling" directory to the "anylabeling" directory.
    """
    print('Migrating anylabeling from "anylabeling" to "anylabeling"...')

    path = Path('anylabeling/labeled')

    next_sample_id = _get_highest_sample_id() + 1

    for item in path.rglob('*.json'):
        if not item.name.startswith('img'):
            print(f'Skipping file "{item.name}". File name needs to start with "img".')
            continue

        data = json.loads(item.read_text())

        if data.get('imagePath') is None:
            print(f'Skipping file "{item.name}". Image path needs to be set.')
            continue

        if not data.get('imagePath').startswith('img'):
            print(f'Skipping file "{item.name}". Image path needs to start with "img".')
            continue

        old_image_path = item.parent / data['imagePath']

        processors_by_version = {
            '0.4.8': _process_polygons_from_version_0_4_8
        }

        version = data.get('version')

        if version not in processors_by_version:
            print(f'Skipping file "{item.name}". Version "{data.get("version")}" is not supported.')
            continue

        sample_path = Path(f'data/anylabeling/{next_sample_id}')
        sample = Sample(sample_path=sample_path)

        try:
            sample.create()
            sample.copy_image_from(old_image_path)

            cv2.imwrite(str(sample_path / 'image.png'), cv2.imread(str(old_image_path)))

            processor = processors_by_version[version]
            processed = processor(data)

            sample.save_polygons(processed)
            sample.add_source('anylabeling')

            (sample_path / 'polygons.json').write_text(json.dumps(processed, indent=4))

            next_sample_id += 1

        except Exception as e:
            print(f'Error while migrating file "{item.name}": {e}')

            for _ in sample_path.rglob('*'):
                _.unlink()
            sample_path.rmdir()

        item.unlink()
        old_image_path.unlink()

        print(f'Migrated file "{item.name}".')
