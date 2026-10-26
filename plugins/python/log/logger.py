# Logger
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import ABC, abstractmethod
import logging


class Logger(ABC):
    """Abstract interface for logging."""

    @abstractmethod
    def error(self, message: str, *args):
        pass

    @abstractmethod
    def warning(self, message: str, *args):
        pass

    @abstractmethod
    def info(self, message: str, *args):
        pass

    @abstractmethod
    def debug(self, message: str, *args):
        pass


class PythonLogger(Logger):
    """Logger implementation using Python logging."""

    def __init__(self):
        logging.basicConfig(level=logging.WARNING)

    def error(self, message, *args):
        logging.error(message % args if args else message)

    def warning(self, message, *args):
        logging.warning(message % args if args else message)

    def info(self, message, *args):
        logging.info(message % args if args else message)

    def debug(self, message, *args):
        logging.debug(message % args if args else message)
