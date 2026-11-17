import numpy as np

from utils.scene_match import similarity_between

MATCH_SCALE = 0.5
CHAIN_INLIERS = 20


def chain(matcher, features, anchor, count):
    transforms = {anchor: np.eye(3)}
    for direction in (-1, 1):
        index = anchor
        while 0 <= index + direction < count:
            step = similarity_between(
                matcher,
                features[index],
                features[index + direction],
                CHAIN_INLIERS,
                match_scale=MATCH_SCALE,
            )
            if step is None:
                break
            transforms[index + direction] = step @ transforms[index]
            index += direction
    return transforms


def crossing(transforms, far, near, boxes, start, frames_per_second):
    leads = {}
    direction = None
    for index in sorted(boxes):
        if index not in transforms:
            continue
        a = transforms[index] @ np.array([*far, 1.0])
        b = transforms[index] @ np.array([*near, 1.0])
        a, b = a[:2], b[:2]
        line = b - a
        length = np.linalg.norm(line)
        normal = np.array([line[1], -line[0]]) / length
        unit = line / length
        distances = []
        for x1, _, x2, y2 in boxes[index]:
            for point in (np.array([x1, y2]), np.array([x2, y2])):
                along = float(np.dot(point - a, unit)) / length
                # bottom corners above the far rail are posters and people behind the rail
                if -0.1 <= along <= 1.4:
                    distances.append(float(np.dot(point - a, normal)))
        if not distances:
            continue
        if direction is None:
            direction = -np.sign(np.median(distances))
        leads[index] = max(direction * d for d in distances)
    indices = sorted(leads)
    if not indices:
        return {"status": "no located frames with racers"}
    if leads[indices[0]] > 0:
        if len(indices) < 2 or leads[indices[1]] <= leads[indices[0]]:
            return {"status": "line located only after the crossing"}
        rate = (leads[indices[1]] - leads[indices[0]]) / (indices[1] - indices[0])
        return {
            "status": "extrapolated",
            "seconds": start
            + (indices[0] - leads[indices[0]] / rate) / frames_per_second,
            "first_located": start + indices[0] / frames_per_second,
        }
    for previous, current in zip(indices, indices[1:]):
        if leads[current] > 0:
            fraction = -leads[previous] / (leads[current] - leads[previous])
            return {
                "status": (
                    "ok" if current - previous == 1 else f"gap {current - previous}"
                ),
                "seconds": start
                + (previous + fraction * (current - previous)) / frames_per_second,
            }
    return {"status": "no crossing in window"}
