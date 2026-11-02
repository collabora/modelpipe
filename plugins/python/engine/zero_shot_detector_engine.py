# ZeroShotDetectorEngine
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .onnx_export import (
    exported_model_path,
    model_input_frames,
    model_input_shape,
    pixel_normalizer,
)
from .pytorch_engine import PyTorchEngine

DEFAULT_CONFIDENCE = 0.1
BATCH_DIMENSIONS = 4
# the epsilon OWL's class head adds to both norms
CLASS_HEAD_NORM_EPSILON = 1e-6
# the families whose image encoder never sees the text
BOX_SCALE_BY_MODEL_TYPE = {
    # owlv2 pads the frame to a square before resizing
    "owlv2": lambda height, width: (max(height, width), max(height, width)),
    "owlvit": lambda height, width: (width, height),
}


def detections(processed, labels):
    boxes = [[float(value) for value in box] for box in processed["boxes"]]
    scores = [float(score) for score in processed["scores"]]
    indices = label_indices(processed, labels)
    kept = [i for i, index in enumerate(indices) if index is not None]
    return {
        "boxes": [boxes[i] for i in kept],
        "labels": [indices[i] for i in kept],
        "scores": [scores[i] for i in kept],
    }


def label_indices(processed, labels):
    text_labels = processed.get("text_labels")
    if text_labels is None:
        return [int(label) for label in processed["labels"]]
    return [index_of(text, labels) for text in text_labels]


def index_of(text, labels):
    # grounding dino reports the phrase it decoded, not an index into the labels
    phrase = text.strip().lower()
    if not phrase:
        return None
    for index, label in enumerate(labels):
        if label.strip().lower() == phrase:
            return index
    for index, label in enumerate(labels):
        if phrase in label.strip().lower():
            return index
    return None


class ExportedZeroShotDetector:
    def __init__(self, model_name, engine_name):
        from transformers import (
            AutoConfig,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        model_type = AutoConfig.from_pretrained(model_name).model_type
        if model_type not in BOX_SCALE_BY_MODEL_TYPE:
            raise ValueError(
                f"only {', '.join(BOX_SCALE_BY_MODEL_TYPE)} models export, "
                f"'{model_name}' is {model_type}, run it on the pytorch engine"
            )
        self.box_scale = BOX_SCALE_BY_MODEL_TYPE[model_type]
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name
        ).eval()
        self.query_embeddings_by_labels = {}
        self.labels = []
        self.confidence = DEFAULT_CONFIDENCE
        self.track = False
        self.engine = None
        self.path = exported_model_path(
            engine_name,
            f"{model_name.replace('/', '--')}-image-detector",
            self._build_image_graph,
        )

    def _build_image_graph(self):
        import torch

        model = self.model
        image_processor = self.processor.image_processor
        normalize = pixel_normalizer(
            image_processor.image_mean, image_processor.image_std
        )

        class ZeroShotImageGraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, image):
                feature_map, _ = self.model.image_embedder(
                    pixel_values=normalize(image)
                )
                batch_size, patch_rows, patch_columns, hidden_size = feature_map.shape
                image_features = feature_map.reshape(
                    batch_size, patch_rows * patch_columns, hidden_size
                )
                class_head = self.model.class_head
                class_embeddings = class_head.dense0(image_features)
                class_embeddings = class_embeddings / (
                    torch.linalg.norm(class_embeddings, dim=-1, keepdim=True)
                    + CLASS_HEAD_NORM_EPSILON
                )
                logit_shift = class_head.logit_shift(image_features)
                logit_scale = class_head.elu(class_head.logit_scale(image_features)) + 1
                boxes = self.model.box_predictor(image_features, feature_map)
                return boxes, class_embeddings, logit_shift, logit_scale

        height, width, channels = model_input_shape(image_processor)
        return ZeroShotImageGraph(), torch.rand(1, channels, height, width)

    def _query_embeddings(self, labels):
        import torch

        key = tuple(labels)
        if key not in self.query_embeddings_by_labels:
            tokens = self.processor(text=[labels], return_tensors="pt", padding=True)
            with torch.no_grad():
                text_embeddings = self.model.base_model.get_text_features(
                    **tokens
                ).pooler_output
            text_embeddings = text_embeddings / torch.linalg.norm(
                text_embeddings, ord=2, dim=-1, keepdim=True
            )
            query_embeddings = text_embeddings / (
                torch.linalg.norm(text_embeddings, dim=-1, keepdim=True)
                + CLASS_HEAD_NORM_EPSILON
            )
            self.query_embeddings_by_labels[key] = query_embeddings.numpy()
        return self.query_embeddings_by_labels[key]

    def do_forward(self, frames):
        if not self.labels:
            raise ValueError("no labels set for zero-shot detection")
        if frames.ndim == BATCH_DIMENSIONS:
            return [self._frame_detections(frame) for frame in frames]
        return self._frame_detections(frames)

    def _frame_detections(self, frame):
        import numpy as np

        model_input = model_input_frames(self.processor.image_processor, frame)
        boxes, class_embeddings, logit_shift, logit_scale = (
            output[0] for output in self.engine.do_forward(model_input)
        )
        query_embeddings = self._query_embeddings(self.labels)
        logits = (class_embeddings @ query_embeddings.T + logit_shift) * logit_scale
        best_labels = logits.argmax(axis=-1)
        scores = 1 / (1 + np.exp(-logits.max(axis=-1)))
        kept = scores > self.confidence
        center_x, center_y, box_width, box_height = boxes[kept].T
        x_scale, y_scale = self.box_scale(*frame.shape[:2])
        corners = np.stack(
            [
                (center_x - box_width / 2) * x_scale,
                (center_y - box_height / 2) * y_scale,
                (center_x + box_width / 2) * x_scale,
                (center_y + box_height / 2) * y_scale,
            ],
            axis=-1,
        )
        processed = {
            "boxes": corners,
            "scores": scores[kept],
            "text_labels": [self.labels[index] for index in best_labels[kept]],
        }
        return detections(processed, self.labels)


class ZeroShotDetectorEngine(PyTorchEngine):
    PRECISIONS = ("fp32", "fp16")
    TORCH_DTYPE_NAMES = {"fp32": "float32", "fp16": "float16"}

    def __init__(self):
        super().__init__()
        self.labels = []
        self.confidence = DEFAULT_CONFIDENCE

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self.image_processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(
                f"Zero-shot detection model '{model_name}' loaded on {self.device}"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load zero-shot detection model '{model_name}': {e}"
            )

    def do_forward(self, frames):
        import numpy as np
        import torch
        from PIL import Image

        if not self.labels:
            raise ValueError("no labels set for zero-shot detection")
        if self.model is None or self.image_processor is None:
            raise ValueError("zero-shot detection model is not loaded")
        dtype = getattr(torch, self.TORCH_DTYPE_NAMES[self.precision])
        if self.model.dtype != dtype:
            self.model.to(dtype)

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        batch = frames if is_batch else np.expand_dims(frames, 0)
        images = [Image.fromarray(frame.astype(np.uint8)) for frame in batch]
        candidate_labels = [self.labels] * len(images)

        inputs = self.image_processor(
            images=images,
            text=candidate_labels,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            outputs = self.model(**inputs)

        processed = self.image_processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.confidence,
            target_sizes=[(image.height, image.width) for image in images],
            text_labels=candidate_labels,
        )
        results = [detections(item, self.labels) for item in processed]
        self.logger.debug(
            f"Zero-shot detections per frame: {[len(r['boxes']) for r in results]}"
        )
        return results if is_batch else results[0]
