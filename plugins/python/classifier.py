# Classifier
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend
from backend import CUDA_INFERENCE_FRAME_CAPABILITIES

CAN_REGISTER_ELEMENT = True
try:
    from base_classifier import BaseClassifier
    from engine.classifier_engine import ClassifierEngine, ExportedClassifier
    from engine.engine_factory import EngineFactory
    from engine.support_matrix import TASK_ENGINES
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'classifier_pylm' element will not be available. Error: {e}"
    )


class Classifier(BaseClassifier):
    """
    GStreamer element for image classification.
    """

    __gstmetadata__ = (
        "Classifier",
        "Transform",
        "Classifier",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )
    FRAME_CAPABILITIES = CUDA_INFERENCE_FRAME_CAPABILITIES

    def __init__(self):
        super().__init__()
        self.task_engine_name = self.mgr.engine_name = "mp_classifier_engine"
        EngineFactory.register(self.mgr.engine_name, ClassifierEngine)

    def export_model(self, model_name):
        engine = self.mgr.engine_name
        if engine not in TASK_ENGINES["classifier"]:
            raise ValueError(f"{engine} does not run the classifier, see mp-engines")
        return ExportedClassifier(model_name, engine)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_classifier", Classifier)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_classifier' element will not be registered because a module is missing."
    )
