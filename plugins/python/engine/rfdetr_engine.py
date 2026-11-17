# RFDETREngine
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

from .pytorch_engine import PyTorchEngine


class RFDETREngine(PyTorchEngine):
    def do_load_model(self, model_name, **kwargs):
        import torch
        from rfdetr import RFDETR
        from rfdetr.utilities.class_names import prediction_labels

        checkpoint = RFDETR.from_checkpoint(model_name, device=self.device)
        self.model = checkpoint.inference(dtype=torch.float16)
        # rfdetr stock COCO checkpoints return sparse COCO category ids, not positions in class_names
        self.categories = prediction_labels(checkpoint).class_id_to_name
        self.logger.info(f"RF-DETR model '{model_name}' loaded on {self.device}")
        return True

    def do_forward(self, frame):
        detections = self.model.predict(
            frame, threshold=self.conf, include_source_image=False
        )
        if self.classes:
            unknown = self.classes - set(self.categories.values())
            if unknown:
                raise ValueError(f"classes {sorted(unknown)} are not in the model")
            kept = [
                self.categories[int(class_id)] in self.classes
                for class_id in detections.class_id
            ]
            detections = detections[np.array(kept, dtype=bool)]
        return {
            "boxes": detections.xyxy,
            "labels": detections.class_id,
            "scores": detections.confidence,
        }
