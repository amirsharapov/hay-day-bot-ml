import sys

from src.features.anylabeling.process import process_labeled_images


def main():
    args = sys.argv[1:]

    visited = set()

    for arg in args:
        if arg in visited:
            continue

        if arg == '--process-labeled-images':
            process_labeled_images()

        visited.add(arg)

if __name__ == '__main__':
    main()
