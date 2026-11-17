import argparse
import collections
import csv
import pathlib

import cv2
import numpy as np

CROPS_PER_RACE = 200
CLUSTERS_PER_RACE = 14
COLUMNS, ROWS = 5, 6
CELL_WIDTH, CELL_HEIGHT = 300, 200
# number cloths and silks sit in the top part of the box, legs and dirt below
UPPER_FRACTION = 0.6


def colour_feature(image):
    upper = image[: int(image.shape[0] * UPPER_FRACTION)]
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    saturated = (hsv[..., 1] > 90) & (hsv[..., 2] > 70)
    histogram = cv2.calcHist(
        [hsv], [0, 1], saturated.astype(np.uint8), [18, 3], [0, 180, 90, 256]
    ).ravel()
    white = ((hsv[..., 1] < 40) & (hsv[..., 2] > 190)).sum()
    counts = np.append(histogram, white)
    return np.sqrt(counts / max(counts.sum(), 1))


def colour_order(images):
    features = np.float32([colour_feature(image) for image in images])
    _, clusters, _ = cv2.kmeans(
        features,
        CLUSTERS_PER_RACE,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    return np.argsort(clusters.ravel(), kind="stable")


def cell(image, number):
    tile = np.full((CELL_HEIGHT, CELL_WIDTH, 3), 255, np.uint8)
    room_height = CELL_HEIGHT - 30
    scale = min((CELL_WIDTH - 4) / image.shape[1], room_height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    height, width = resized.shape[:2]
    left = (CELL_WIDTH - width) // 2
    tile[2 : 2 + height, left : left + width] = resized
    cv2.putText(
        tile,
        str(number),
        (4, CELL_HEIGHT - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    return tile


def main():
    parser = argparse.ArgumentParser(
        description=f"Contact sheets of {COLUMNS * ROWS} numbered horse crops, "
        f"{CROPS_PER_RACE} per race in colour cluster order"
    )
    parser.add_argument(
        "crops", type=pathlib.Path, help="output of extract_horse_crops.py"
    )
    parser.add_argument("output", type=pathlib.Path, help="directory for the sheets")
    arguments = parser.parse_args()
    by_race = collections.defaultdict(list)
    for row in csv.DictReader(open(arguments.crops / "crops.csv")):
        by_race[row["clip"]].append(row["crop"])
    random = np.random.default_rng(0)
    cv2.setRNGSeed(0)
    ordered = []
    for race in sorted(by_race):
        names = random.choice(by_race[race], CROPS_PER_RACE, replace=False)
        images = [cv2.imread(str(arguments.crops / f"{name}.jpg")) for name in names]
        ordered += [(names[index], images[index]) for index in colour_order(images)]
    arguments.output.mkdir(parents=True, exist_ok=True)
    per_sheet = COLUMNS * ROWS
    for sheet, start in enumerate(range(0, len(ordered), per_sheet)):
        batch = ordered[start : start + per_sheet]
        tiles = [cell(image, number) for number, (_, image) in enumerate(batch, 1)]
        tiles += [np.full_like(tiles[0], 255)] * (per_sheet - len(tiles))
        image = np.vstack(
            [np.hstack(tiles[r * COLUMNS : (r + 1) * COLUMNS]) for r in range(ROWS)]
        )
        cv2.imwrite(str(arguments.output / f"{sheet:04d}.png"), image)
        with open(arguments.output / f"{sheet:04d}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["number", "crop"])
            writer.writerows(
                (number, name) for number, (name, _) in enumerate(batch, 1)
            )
    print(sheet + 1, "sheets")


if __name__ == "__main__":
    main()
