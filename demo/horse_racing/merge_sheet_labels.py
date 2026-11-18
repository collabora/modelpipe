import argparse
import collections
import csv
import pathlib

from racer_number_classifier import racer_number_classes

HIGHEST_NUMBER = 14
SHEET_CLASSES = {
    "numbers": racer_number_classes(HIGHEST_NUMBER),
    "other": ["other", "gate_scene", "horses_running"],
}


def read_sheet_labels(path, classes):
    labels = {}
    for row in csv.DictReader(open(path)):
        if row["class"] not in classes:
            raise ValueError(f"{path}: unknown class {row['class']!r}")
        labels[(row["sheet"], int(row["number"]))] = row["class"]
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Join the agent sheet labels to the sheet images and score "
        "their agreement"
    )
    parser.add_argument("kind", choices=sorted(SHEET_CLASSES))
    parser.add_argument(
        "sheets",
        type=pathlib.Path,
        help="output of number_sheets.py or frame_sheets.py",
    )
    parser.add_argument("labels", type=pathlib.Path, help="folder of agent csv files")
    arguments = parser.parse_args()
    classes = SHEET_CLASSES[arguments.kind]
    first = {}
    for path in sorted(arguments.labels.glob("batch_*.csv")):
        first |= read_sheet_labels(path, classes)
    second = read_sheet_labels(arguments.labels / "second.csv", classes)
    image_of = {}
    for path in sorted(arguments.sheets.glob("*.csv")):
        for row in csv.DictReader(open(path)):
            image_of[(path.stem, int(row["number"]))] = row["image"]
    missing = image_of.keys() - first.keys()
    if missing:
        raise ValueError(f"{len(missing)} images have no label, such as {min(missing)}")
    agreed = sum(first[key] == label for key, label in second.items())
    print(f"agreement {agreed}/{len(second)} = {agreed / len(second):.3f}")
    disagreements = collections.Counter(
        (first[key], label) for key, label in second.items() if first[key] != label
    )
    for (first_label, second_label), count in disagreements.most_common():
        print(f"  first {first_label} second {second_label}: {count}")
    class_of = {image_of[key]: first[key] for key in image_of}
    print(collections.Counter(class_of.values()).most_common())
    with open(arguments.labels / "labels.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "class"])
        writer.writerows(sorted(class_of.items()))


if __name__ == "__main__":
    main()
