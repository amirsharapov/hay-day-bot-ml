import sys

from src.process_anylabeling_data import process_anylabeling_data


def main():
    args = sys.argv[1:]
    args = iter(args)

    visited = set()

    for arg in args:
        if arg in visited:
            continue

        if arg == '--process-anylabeling-data':
            name = next(args)
            process_anylabeling_data(name)

        visited.add(arg)

if __name__ == '__main__':
    # main()
    process_anylabeling_data('datasets/chickens_ready_for_harvest')
