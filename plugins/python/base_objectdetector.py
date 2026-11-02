# object_detector_base.py
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from video_transform import VideoTransform
from utils.format_converter import FormatConverter
from backend import analytics, GObject
from tasks.object_detector import ObjectDetectorTask

LAST_FRAME_DEFAULT = 2**31 - 1


class BaseObjectDetector(VideoTransform, ObjectDetectorTask):
    """
    GStreamer element shell for object detection with batch processing support.
    Handles both single-frame buffers (no metadata) and batch buffers (metadata
    in last chunk). The inference and metadata steps (do_forward / do_decode)
    are inherited from the backend-agnostic ObjectDetectorTask; this class only
    supplies the GStreamer per-buffer glue.
    """

    def __init__(self):
        super().__init__()
        self.framerate_num = 30
        self.framerate_denom = 1
        self.format_converter = FormatConverter()
        self.logger.info("Initialized BaseObjectDetector")
        self.__track = False
        self.__interval = 1
        self.__first_frame = 0
        self.__last_frame = LAST_FRAME_DEFAULT
        self._frame_number = 0
        self._det_counter = 0
        self._cached_results = None
        self._cached_num_sources = 1

    @GObject.Property(type=bool, default=False)
    def track(self):
        "Enable or disable tracking mode"
        return self.__track

    @track.setter
    def track(self, value):
        self.__track = value
        if self.engine:
            self.engine.track = value

    @GObject.Property(type=int, default=1, minimum=1, maximum=10000)
    def interval(self):
        "Run detection every Nth frame and re-attach the previous detections on"
        "the frames in between (N=1 runs detection every frame). Lets downstream "
        "tracking/overlay stay per-frame while detection runs at a lower rate."
        return self.__interval

    @interval.setter
    def interval(self, value):
        self.__interval = max(1, int(value))

    @GObject.Property(
        type=int,
        default=0,
        minimum=0,
        maximum=LAST_FRAME_DEFAULT,
        nick="First Frame",
        blurb="First frame, counted from zero, to run detection on",
        flags=GObject.ParamFlags.READWRITE,
    )
    def first_frame(self):
        return self.__first_frame

    @first_frame.setter
    def first_frame(self, value):
        self.__first_frame = value

    @GObject.Property(
        type=int,
        default=LAST_FRAME_DEFAULT,
        minimum=0,
        maximum=LAST_FRAME_DEFAULT,
        nick="Last Frame",
        blurb="Last frame, counted from zero, to run detection on. Frames "
        "outside first-frame to last-frame pass through with no detections.",
        flags=GObject.ParamFlags.READWRITE,
    )
    def last_frame(self):
        return self.__last_frame

    @last_frame.setter
    def last_frame(self, value):
        self.__last_frame = value

    @GObject.Property(type=str, default="30/1")
    def framerate(self):
        "Source framerate as 'num/denom', used for muxed-stream frame timing"
        return f"{self.framerate_num}/{self.framerate_denom}"

    @framerate.setter
    def framerate(self, value):
        try:
            num, denom = str(value).split("/")
            self.framerate_num = int(num)
            self.framerate_denom = int(denom)
        except (ValueError, AttributeError):
            self.logger.warning(f"Invalid framerate '{value}', expected 'num/denom'")

    def process_frames(self, frames, num_sources, fmt, target):
        """
        Run detection on the extracted frame(s) and attach results to `target`.

        Backend-agnostic per-frame hook: frame extraction and the FlowReturn
        wrapping live in the backend driver (the gst `do_transform_ip` /
        the g2g `g2g_process`), so the inference + metadata here is identical on
        every backend. Raises on a hard failure; the driver maps that to its own
        error return.
        """
        frame_number = self._frame_number
        self._frame_number += 1
        if not self.__first_frame <= frame_number <= self.__last_frame:
            return
        run_detect = (self._det_counter % self.__interval) == 0
        self._det_counter += 1

        if run_detect:
            results = self.do_forward(frames)
            if results is None:
                raise RuntimeError("inference returned None")
            self._cached_results = results
            self._cached_num_sources = num_sources
            self._decode_results(target, results, num_sources)
        elif self._cached_results is not None:
            # Skip inference on this frame and re-attach the previous
            # detections so downstream tracking/overlay stay per-frame.
            self._decode_results(target, self._cached_results, self._cached_num_sources)

        attached_meta = analytics.get_relation_meta(target)
        if attached_meta:
            count = analytics.relation_length(attached_meta)
            self.logger.info(f"Total metadata relations attached: {count}")
        else:
            self.logger.debug("No detections on this buffer")

    def _decode_results(self, target, results, num_sources):
        # Single-frame case
        if num_sources == 1:
            self.do_decode(target, results, stream_idx=0)
        # Batch case
        else:
            self.logger.info(f"Processing batch with num_sources={num_sources}")
            results_list = results if isinstance(results, list) else [results]
            if len(results_list) != num_sources:
                raise RuntimeError(
                    f"expected {num_sources} results, got {len(results_list)}"
                )
            for idx, result in enumerate(results_list):
                if result is None:
                    self.logger.warning(f"Frame {idx} result is None")
                    continue
                self.do_decode(target, result, stream_idx=idx)
