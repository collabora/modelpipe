# RacerNumberClassifier
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json

import backend
from backend import analytics, frameio, GObject, InPlaceVideoTransform
from backend.video_frame import (
    RGBA_PIXEL_FORMATS,
    SYSTEM_MEMORY,
    VideoFrameCapabilities,
)
from utils.blob_headers import (
    RACE_WINDOW_BLOB_NAME,
    RACER_NUMBER_PROBABILITIES_META_HEADER,
)
from utils.image_classifier import load_image_classifier

NO_NUMBER_CLASS = "none"
RACER_NUMBER_COUNT = 8
RACER_NUMBER_CLASSES = [str(number) for number in range(1, RACER_NUMBER_COUNT + 1)] + [
    NO_NUMBER_CLASS
]


CLASSIFIER_DEVICE = "cuda:0"


class RacerNumberClassifier(InPlaceVideoTransform):
    __gstmetadata__ = (
        "Racer Number Classifier",
        "Filter/Analyzer/Video",
        "On frames carrying a race-window blob, classifies the number cloth "
        "colour of each racer box and writes a racer-number-probabilities blob "
        "with the probability of racer numbers 1 to 8 and of no number for "
        "each box",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    FRAME_CAPABILITIES = VideoFrameCapabilities(
        (SYSTEM_MEMORY,), RGBA_PIXEL_FORMATS, True, True
    )

    model = GObject.Property(
        type=str,
        default="",
        nick="Model",
        blurb="torchvision classifier of racer crops into racer numbers 1 to 8 "
        "and none",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self._classifier = None

    def process_in_place(self, frame, fmt, buf):
        if RACE_WINDOW_BLOB_NAME not in frameio.read_blobs(buf):
            return
        meta = analytics.get_relation_meta(buf)
        if not meta:
            return
        racers = analytics.read_objects(meta)
        if not racers:
            return
        if self._classifier is None:
            if not self.model:
                raise ValueError("set model")
            self._classifier = load_image_classifier(self.model, CLASSIFIER_DEVICE)
            if self._classifier.classes != RACER_NUMBER_CLASSES:
                raise ValueError(f"{self.model} classes {self._classifier.classes}")
        crops = [
            frame[
                int(o["y"]) : int(o["y"] + o["h"]),
                int(o["x"]) : int(o["x"] + o["w"]),
                :3,
            ]
            for o in racers
        ]
        probabilities = self._classifier.probabilities(crops)
        number_probabilities = {
            str(racer["ref"]): racer_probabilities
            for racer, racer_probabilities in zip(racers, probabilities.tolist())
        }
        frameio.append_blob(
            buf,
            RACER_NUMBER_PROBABILITIES_META_HEADER,
            json.dumps(number_probabilities).encode(),
        )


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_racer_number_classifier", RacerNumberClassifier
    )
