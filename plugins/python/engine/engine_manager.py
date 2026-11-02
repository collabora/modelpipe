# EngineManager
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import traceback

from .engine_factory import EngineFactory


class EngineManager:
    def __init__(self, logger, default_engine=EngineFactory.PYTORCH_ENGINE):
        self.logger = logger
        self.engine_name = default_engine
        self.engine = None
        self.device = "cpu"  # Manage device here

    def initialize_engine(self):
        if not self.engine and self.engine_name:
            engine = EngineFactory.create(self.engine_name)
            engine.do_set_device(self.device)
            self.engine = engine
        if not self.engine:
            self.logger.error(f"Unable to load ML engine: {self.engine_name}")

    def set_device(self, device):
        self.device = device
        if self.engine:
            # pygobject swallows a raise in a property setter
            engine, self.engine = self.engine, None
            engine.do_set_device(device)
            self.engine = engine

    def do_load_model(self, model_name, **kwargs):
        self.initialize_engine()
        if self.engine is None:
            self.logger.warning(
                f"Cannot load model {model_name}: engine not initialized"
            )
            return False
        if self.engine.model:
            return True
        if model_name is None:
            stack_trace = (
                traceback.format_stack()
            )  # Capture the current call stack as a list of strings
            self.logger.warning(
                f"Cannot load model as model name is not set\n"
                f"Stack trace:\n{''.join(stack_trace)}"  # Join and log the stack trace
            )
            return False
        if self.engine.precision not in self.engine.PRECISIONS:
            raise ValueError(
                f"{type(self.engine).__name__} runs "
                f"{', '.join(self.engine.PRECISIONS)}, not precision="
                f"{self.engine.precision}"
            )
        try:
            self.logger.info(f"Loading model: {model_name}")
            self.engine.do_load_model(model_name, **kwargs)
            self.logger.info(f"Model {model_name} loaded successfully")
            return True
        except Exception:
            self.engine.tokenizer = None
            self.engine.model = None
            # an engine reloads the model it names when its device changes
            self.engine.model_name = None
            raise

    def get_model(self):
        self.initialize_engine()
        if self.engine is None:
            raise ValueError("Engine is not present, unable to get model")
        return self.engine.get_model()

    def get_tokenizer(self):
        if self.engine:
            model = self.get_model()
            if model is None:
                self.logger.warning("Model not loaded yet, attempting to load.")
            return self.engine.tokenizer
        else:
            self.logger.warning("Engine is not present, unable to get the tokenizer.")
            return None
