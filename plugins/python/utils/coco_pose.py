# COCO pose
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from functools import cache

# ultralytics draws a keypoint from this confidence
DRAWN_KEYPOINT_CONFIDENCE = 0.25


# keypoint index pairs joined by a limb, in ultralytics' limb colour order
@cache
def coco_skeleton():
    import numpy as np
    from ultralytics.utils.plotting import Annotator

    annotator = Annotator(np.zeros((1, 1, 3), np.uint8))
    return [(first - 1, second - 1) for first, second in annotator.skeleton]
