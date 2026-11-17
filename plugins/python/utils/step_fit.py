# step fit
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

EDGE_FRAMES = 5


def step_fit(after):
    import numpy as np

    after = np.asarray(after)
    split = max(
        range(EDGE_FRAMES, len(after) - EDGE_FRAMES),
        key=lambda k: after[k:].mean() - after[:k].mean(),
    )
    return split, float(after[split:].mean() - after[:split].mean())
