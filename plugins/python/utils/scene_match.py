# scene matching
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

RACER_MASK_MARGIN = 10
SIFT_FEATURES = 3000
# lowe's ratio test
MATCH_RATIO = 0.75
RANSAC_THRESHOLD_PIXELS = 2.0


def matching_features(sift, gray, racer_boxes, match_scale):
    import cv2
    import numpy as np

    small = cv2.resize(gray, None, fx=match_scale, fy=match_scale)
    mask = np.full(small.shape[:2], 255, np.uint8)
    for x1, y1, x2, y2 in racer_boxes:
        cv2.rectangle(
            mask,
            (
                int(x1 * match_scale) - RACER_MASK_MARGIN,
                int(y1 * match_scale) - RACER_MASK_MARGIN,
            ),
            (
                int(x2 * match_scale) + RACER_MASK_MARGIN,
                int(y2 * match_scale) + RACER_MASK_MARGIN,
            ),
            0,
            -1,
        )
    return sift.detectAndCompute(small, mask)


def similarity_between(
    matcher, source_features, target_features, minimum_inliers, match_scale
):
    import cv2
    import numpy as np

    source_points, source_descriptors = source_features
    target_points, target_descriptors = target_features
    if source_descriptors is None or target_descriptors is None:
        return None
    if len(source_descriptors) < 2 or len(target_descriptors) < 2:
        return None
    pairs = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance
    ]
    if len(good) < minimum_inliers:
        return None
    source = np.float32([source_points[match.queryIdx].pt for match in good])
    target = np.float32([target_points[match.trainIdx].pt for match in good])
    # a panning and zooming camera moves the background by a similarity
    similarity, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD_PIXELS
    )
    if similarity is None or int(inliers.sum()) < minimum_inliers:
        return None
    # the rotation and scale part is the same at full size, the shift grows with it
    similarity[:, 2] /= match_scale
    return np.vstack([similarity, [0, 0, 1]])


def transform_points(transform, points):
    import numpy as np

    homogeneous = np.hstack([np.asarray(points, float), np.ones((len(points), 1))])
    return (homogeneous @ transform.T)[:, :2]
