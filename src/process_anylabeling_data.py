import json

from src.datasets import Dataset


def process_anylabeling_data(dataset_name):
    dataset = Dataset(dataset_name)

    for label_file in dataset.anylabeling_dir.glob('*.json'):
        image_file = label_file.with_suffix('.png')

        sample = dataset.create_sample_dir()

        try:
            new_data = {
                'polygons': []
            }

            old_data = label_file.read_text()
            old_data = json.loads(old_data)

            for shape in old_data['shapes']:
                label = shape['label']
                points = [[int(x), int(y)] for x, y in shape['points']]

                new_data['polygons'].append({
                    'label': label,
                    'points': points,
                    'source': 'anylabeling'
                })

            sample.write_polygons(new_data)
            sample.copy_image_from(image_file)

        except Exception as e:
            print(f'Failed to process "{label_file.name}": {e}')
            sample.destroy()
            continue



