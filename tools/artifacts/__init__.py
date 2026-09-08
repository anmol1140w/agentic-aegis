"""Deterministic local office-artifact generation for AEGIS."""

from .common import ArtifactResult, ArtifactError, resolve_workspace_path
from .service import ArtifactService

__all__ = ["ArtifactError", "ArtifactResult", "ArtifactService", "resolve_workspace_path"]
