"""Structured runtime errors used by providers and the orchestrator."""

from __future__ import annotations


class WorkbenchError(Exception):
    """Base class for errors that are safe to show in the terminal and trace."""


class ModelUnavailableError(WorkbenchError):
    pass


class ModelTimeoutError(WorkbenchError):
    pass


class ProviderError(WorkbenchError):
    pass


class EmptyGenerationError(WorkbenchError):
    pass


class InvalidImageError(WorkbenchError):
    pass


class InvalidOCRInputError(WorkbenchError):
    pass


class OCRBackendError(WorkbenchError):
    pass


class OutputPersistenceError(WorkbenchError):
    pass
