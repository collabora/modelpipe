# RFDETREngine
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .pytorch_engine import PyTorchEngine


def rfdetr_predictor(checkpoint_path, device):
    import torch
    from rfdetr import RFDETR

    checkpoint = RFDETR.from_checkpoint(checkpoint_path, device=device)
    return checkpoint.inference(dtype=torch.float16)


class RFDETREngine(PyTorchEngine):
    def do_load_model(self, model_name, **kwargs):
        self.model = rfdetr_predictor(model_name, self.device)
        self.categories = self.model.class_names
        self.logger.info(f"RF-DETR model '{model_name}' loaded on {self.device}")
        return True

    def do_forward(self, frame):
        detections = self.model.predict(
            frame, threshold=self.conf, include_source_image=False
        )
        return {
            "boxes": detections.xyxy,
            "labels": detections.class_id,
            "scores": detections.confidence,
        }
