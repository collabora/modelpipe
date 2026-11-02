# MLEngine
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import hashlib
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path

from log.logger_factory import LoggerFactory

TORCHVISION_WEIGHTS = "DEFAULT"
TORCHVISION_RESNET_MODULE = "torchvision.models.resnet"
MODEL_CACHE = Path.home() / ".cache" / "modelpipe"
CONVERSION_DIGEST_LENGTH = 12


# the file behind a path can change
def converted_model_path(engine_name, source_path, suffix, *options):
    source = Path(source_path).resolve()
    stat = source.stat()
    key = repr((str(source), stat.st_size, stat.st_mtime_ns, options))
    digest = hashlib.sha256(key.encode()).hexdigest()[:CONVERSION_DIGEST_LENGTH]
    return MODEL_CACHE / engine_name / f"{source.stem}-{digest}{suffix}"


def is_torchvision_resnet(model_name):
    from torchvision import models

    builder = models.get_model_builder(model_name)
    return builder.__module__ == TORCHVISION_RESNET_MODULE


# None when a side of a channels first input is dynamic
def fixed_height_width(shape):
    if len(shape) != 4:
        return None
    height, width = shape[2], shape[3]
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return (height, width)
    return None


class MLEngine(ABC):
    """Abstract base class for machine learning engines that load models, run inference on image frames,
    and generate text with language models."""

    PRECISIONS = ("fp32",)

    def __init__(self):
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.device = None
        self.device_index = 0
        self.model = None
        self.model_name = None
        self.categories = None
        self.tokenizer = None
        self.image_processor = None
        self.batch_size = 1  # Default batch size
        self.frame_buffer = []  # For vision-text models or manual buffering
        self.frame_stride = None
        self.counter = 0
        self.stream_priority = 0
        self.precision = "fp32"
        self.track = False
        self.prompt = "What is shown in this image?"  # Default prompt
        self.input_format = "auto"  # "auto", "nhwc", "nchw"
        self.post_process = (
            "auto"  # "auto", "none", or a format key from detection_decoder
        )

    # Interface #
    @abstractmethod
    def do_load_model(self, model_name, **kwargs):
        """Load a model by name or path, with additional options."""
        pass

    def do_set_device(self, device):
        self.device = device

    def do_forward(self, frames):
        raise NotImplementedError(f"{type(self).__name__} runs no frames")

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise NotImplementedError(f"{type(self).__name__} generates no text")

    def stream_context(self):
        return nullcontext()

    # Implementation #
    def _model_input_hw(self):
        return None

    def _letterbox(self, frames, is_batch):
        """Resize frame(s) to the model input size, preserving aspect ratio with
        grey padding (YOLO-style). Returns (processed, transform); transform =
        (ratio, pad_x, pad_y, orig_w, orig_h) maps model coords back to the
        original frame. Returns (frames, None) when no resize is needed (already
        model-sized, or dynamic input) -- so pre-sized callers are unaffected."""
        import numpy as np
        import cv2

        mhw = self._model_input_hw()
        if mhw is None:
            return frames, None
        mh, mw = mhw
        imgs = frames if is_batch else frames[None]
        h, w = int(imgs.shape[1]), int(imgs.shape[2])
        if (h, w) == (mh, mw):
            return frames, None
        r = min(mh / h, mw / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        pad_x, pad_y = (mw - nw) // 2, (mh - nh) // 2
        out = np.full((imgs.shape[0], mh, mw, imgs.shape[3]), 114, dtype=imgs.dtype)
        for i in range(imgs.shape[0]):
            out[i, pad_y : pad_y + nh, pad_x : pad_x + nw] = cv2.resize(
                imgs[i], (nw, nh), interpolation=cv2.INTER_LINEAR
            )
        proc = out if is_batch else out[0]
        return proc, (r, float(pad_x), float(pad_y), w, h)

    def _unletterbox(self, results, transform):
        """Map detection boxes from model coords back to original-frame coords."""
        import numpy as np

        r, pad_x, pad_y, ow, oh = transform
        for res in results if isinstance(results, list) else [results]:
            if not isinstance(res, dict):
                continue
            b = res.get("boxes")
            if b is None or len(b) == 0:
                continue
            b = np.asarray(b, dtype=np.float32).copy()
            b[:, [0, 2]] = ((b[:, [0, 2]] - pad_x) / r).clip(0, ow)
            b[:, [1, 3]] = ((b[:, [1, 3]] - pad_y) / r).clip(0, oh)
            res["boxes"] = b

    def _apply_input_format(self, img, is_batch):
        """Normalize input to (B, ?, H, W) or (B, H, W, C) per self.input_format."""
        import numpy as np

        if not is_batch:
            img = np.expand_dims(img, axis=0)
        if self.input_format == "nchw":
            img = np.transpose(img, (0, 3, 1, 2))
        # "nhwc" or "auto" → leave as (B, H, W, C)
        return img

    def _top_classes(self, preds, is_batch):
        import numpy as np

        probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
        top_classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        results = [
            {"labels": [int(c)], "scores": [float(s)]}
            for c, s in zip(top_classes, confidences)
        ]
        return results[0] if not is_batch else results

    def _apply_post_process(self, raw, is_batch):
        """Apply post-processing to raw engine output per self.post_process."""
        import numpy as np

        pp = self.post_process
        if pp == "auto":
            if (
                isinstance(raw, np.ndarray)
                and raw.ndim == 3
                and raw.shape[1] >= 5
                and raw.shape[2] > raw.shape[1]
            ):
                pp = "anchor_free"
            else:
                pp = "none"
        if pp != "none" and not isinstance(raw, list):
            from utils.detection_decoder import decode

            results = decode(
                raw,
                pp,
                conf_threshold=getattr(self, "conf", 0.25),
                iou_threshold=getattr(self, "iou", 0.45),
            )
            return results[0] if not is_batch else results
        return raw

    def set_prompt(self, prompt):
        """Set the custom prompt for generating responses."""
        self.prompt = prompt

    def get_prompt(self):
        """Return the custom prompt."""
        return self.prompt

    def get_device(self):
        """Return the device the model is running on."""
        return self.device

    def get_model(self):
        """Return the loaded model for use in inference."""
        return self.model
