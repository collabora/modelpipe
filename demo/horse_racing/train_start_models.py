import argparse
import csv
import dataclasses
import pathlib

import numpy as np

from utils.image_classifier import load_image_classifier, save_image_classifier
from utils.image_classifier_training import (
    DEVICE,
    path_probabilities,
    train_image_classifier,
)

ARCHITECTURE = "efficientnet_b0"
GATE_SCENE_SECONDS = (-5.0, -0.5)
RACE_RUNNING_SECONDS = (0.0, 15.0)
# door frames read from the first horse out lag the doors by a few frames
WINDOW_GAP_SECONDS = 0.15


def state_class(seconds):
    if GATE_SCENE_SECONDS[0] <= seconds < GATE_SCENE_SECONDS[1]:
        return "gate_scene"
    if RACE_RUNNING_SECONDS[0] <= seconds < RACE_RUNNING_SECONDS[1]:
        return "race_running"
    return None


def window_class(seconds):
    if seconds <= -WINDOW_GAP_SECONDS:
        return "before"
    if seconds >= WINDOW_GAP_SECONDS:
        return "after"
    return None


@dataclasses.dataclass(frozen=True)
class StartModel:
    class_of_seconds: object
    image_size: int
    epochs: int


START_MODELS = {
    "state": StartModel(state_class, image_size=256, epochs=10),
    "window": StartModel(window_class, image_size=512, epochs=15),
}


def main():
    parser = argparse.ArgumentParser(
        description="Train the race state or start window classifier of "
        "mp_race_start and print its confusion matrix on the held out races"
    )
    parser.add_argument("kind", choices=sorted(START_MODELS))
    parser.add_argument(
        "frames", type=pathlib.Path, help="output of extract_start_frames.py"
    )
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--test-races", nargs="+", required=True)
    arguments = parser.parse_args()
    start_model = START_MODELS[arguments.kind]
    labelled = []
    for row in csv.DictReader(open(arguments.frames / "frames.csv")):
        if row["kind"] != arguments.kind:
            continue
        name = start_model.class_of_seconds(float(row["seconds_from_start"]))
        if name is not None:
            labelled.append((row["clip"], arguments.frames / row["path"], name))
    classes = sorted({name for _, _, name in labelled})
    split = {"train": [], "test": []}
    for clip, path, name in labelled:
        part = "test" if clip in arguments.test_races else "train"
        split[part].append((path, classes.index(name)))
    print(classes, {part: len(samples) for part, samples in split.items()})
    network = train_image_classifier(
        split["train"],
        len(classes),
        ARCHITECTURE,
        start_model.image_size,
        start_model.epochs,
    )
    save_image_classifier(
        arguments.model, network, ARCHITECTURE, classes, start_model.image_size
    )
    paths, truths = zip(*split["test"])
    predicted = path_probabilities(
        load_image_classifier(arguments.model, DEVICE), list(paths)
    ).argmax(1)
    matrix = np.zeros((len(classes), len(classes)), int)
    np.add.at(matrix, (np.array(truths), predicted), 1)
    print("accuracy", round(np.trace(matrix) / matrix.sum(), 3), "of", matrix.sum())
    print("truth \\ predicted", *classes, sep="\t")
    for name, row in zip(classes, matrix):
        print(name, *row, sep="\t")


if __name__ == "__main__":
    main()
