import argparse
import collections
import csv
import pathlib

from racer_number_classifier import racer_number_classes

HIGHEST_NUMBER = 14
CLASSES = racer_number_classes(HIGHEST_NUMBER)


def read_sheet_labels(path):
    labels = {}
    for row in csv.DictReader(open(path)):
        if row["class"] not in CLASSES:
            raise ValueError(f"{path}: unknown class {row['class']!r}")
        labels[(row["sheet"], int(row["number"]))] = row["class"]
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Join the agent sheet labels to crops and score their agreement"
    )
    parser.add_argument("sheets", type=pathlib.Path, help="output of number_sheets.py")
    parser.add_argument("labels", type=pathlib.Path, help="folder of agent csv files")
    arguments = parser.parse_args()
    first = {}
    for path in sorted(arguments.labels.glob("batch_*.csv")):
        first |= read_sheet_labels(path)
    second = read_sheet_labels(arguments.labels / "second.csv")
    crop_of = {}
    for path in sorted(arguments.sheets.glob("*.csv")):
        for row in csv.DictReader(open(path)):
            crop_of[(path.stem, int(row["number"]))] = row["crop"]
    missing = crop_of.keys() - first.keys()
    if missing:
        raise ValueError(f"{len(missing)} crops have no label, such as {min(missing)}")
    agreed = sum(first[key] == label for key, label in second.items())
    print(f"agreement {agreed}/{len(second)} = {agreed / len(second):.3f}")
    disagreements = collections.Counter(
        (first[key], label) for key, label in second.items() if first[key] != label
    )
    for (first_label, second_label), count in disagreements.most_common():
        print(f"  first {first_label} second {second_label}: {count}")
    class_of = {crop_of[key]: first[key] for key in crop_of}
    print(collections.Counter(class_of.values()).most_common())
    with open(arguments.labels / "labels.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["crop", "class"])
        writer.writerows(sorted(class_of.items()))


if __name__ == "__main__":
    main()
