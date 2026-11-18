import argparse
import pathlib

import cv2
import numpy as np

from number_sheets import write_sheets


def main():
    parser = argparse.ArgumentParser(
        description="Contact sheets of numbered frames drawn at random from a directory"
    )
    parser.add_argument("frames", type=pathlib.Path, help="directory of .jpg frames")
    parser.add_argument("count", type=int)
    parser.add_argument("output", type=pathlib.Path, help="directory for the sheets")
    arguments = parser.parse_args()
    paths = sorted(arguments.frames.glob("*.jpg"))
    picked = np.random.default_rng(0).choice(paths, arguments.count, replace=False)
    write_sheets(
        [(path.stem, cv2.imread(str(path))) for path in sorted(picked)],
        arguments.output,
    )


if __name__ == "__main__":
    main()
