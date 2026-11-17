import argparse
import csv
import pathlib

import numpy as np

from merge_number_labels import CLASSES
from utils.image_classifier import load_image_classifier, save_image_classifier
from utils.image_classifier_training import (
    DEVICE,
    path_probabilities,
    train_image_classifier,
)

ARCHITECTURE = "resnet18"
IMAGE_SIZE = 128
EPOCHS = 40


def main():
    parser = argparse.ArgumentParser(
        description="Train the number cloth classifier and print its confusion "
        "matrix on the held out races"
    )
    parser.add_argument(
        "crops", type=pathlib.Path, help="output of extract_horse_crops.py"
    )
    parser.add_argument(
        "labels", type=pathlib.Path, help="labels.csv of merge_number_labels.py"
    )
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--test-races", nargs="+", required=True)
    arguments = parser.parse_args()
    race_of = {
        row["crop"]: row["clip"]
        for row in csv.DictReader(open(arguments.crops / "crops.csv"))
    }
    split = {"train": [], "test": []}
    for row in csv.DictReader(open(arguments.labels)):
        name = "test" if race_of[row["crop"]] in arguments.test_races else "train"
        split[name].append(
            (arguments.crops / f"{row['crop']}.jpg", CLASSES.index(row["class"]))
        )
    print({name: len(samples) for name, samples in split.items()})
    network = train_image_classifier(
        split["train"], len(CLASSES), ARCHITECTURE, IMAGE_SIZE, EPOCHS
    )
    save_image_classifier(arguments.model, network, ARCHITECTURE, CLASSES, IMAGE_SIZE)
    paths, classes = zip(*split["test"])
    predicted = path_probabilities(
        load_image_classifier(arguments.model, DEVICE), list(paths)
    ).argmax(1)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), int)
    np.add.at(matrix, (np.array(classes), predicted), 1)
    print("accuracy", round(np.trace(matrix) / matrix.sum(), 3), "of", matrix.sum())
    print("truth \\ predicted", *CLASSES, sep="\t")
    for name, row in zip(CLASSES, matrix):
        if row.any():
            print(name, *row, sep="\t")


if __name__ == "__main__":
    main()
