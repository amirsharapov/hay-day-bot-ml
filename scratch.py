from pathlib import Path

for item in Path('datasets/chickens_ready_for_harvest/anylabeling').iterdir():
    if item.name.startswith('sample_000'):
        item.rename(item.name.replace('sample_000', 'img_'))
