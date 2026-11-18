#!/usr/bin/env python3
# Horse racing demo model download
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

from huggingface_hub import hf_hub_download

REPO_ID = "boxerab/modelpipe-horse-racing"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "horse_racing"
)
MODEL_FILES = ("race_state.pth", "race_start_window.pth", "numbers.pth")


def main():
    for name in MODEL_FILES:
        print(hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=LOCAL_DIR))


if __name__ == "__main__":
    main()
