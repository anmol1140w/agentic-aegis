"""Model scoring — rank candidate models for a given task.

The scorer turns a (Task, ModelConfig) pair into a numeric score.
Higher = better fit.  The router calls this for every candidate and
picks the highest scorer.
"""

from __future__ import annotations

from models.base import ModelCapability, ModelConfig
from routing.classifier import Complexity, Modality, Task, TaskType

# ---------------------------------------------------------------------------
# Capability requirements per task type
# ---------------------------------------------------------------------------

_TASK_CAPABILITIES: dict[TaskType, list[ModelCapability]] = {
    TaskType.GENERAL:              [ModelCapability.GENERAL],
    TaskType.CODING:               [ModelCapability.CODING],
    TaskType.DEBUGGING:            [ModelCapability.DEBUGGING],
    TaskType.DOCUMENT_ANALYSIS:    [ModelCapability.DOCUMENT_ANALYSIS],
    TaskType.SUMMARIZATION:        [ModelCapability.SUMMARIZATION],
    TaskType.CALCULATION:          [ModelCapability.CALCULATION],
    TaskType.MULTIMODAL:           [ModelCapability.MULTIMODAL],
    TaskType.IMAGE_ANALYSIS:       [ModelCapability.IMAGE_ANALYSIS, ModelCapability.VISION],
    TaskType.ENGINEERING_DOCUMENT: [ModelCapability.ENGINEERING_DOCUMENT, ModelCapability.VISION],
    TaskType.ARTIFACT_GENERATION:  [ModelCapability.ARTIFACT_GENERATION],
}

# Weights for different scoring dimensions
_W_CAPABILITY = 50     # per matching capability
_W_PRIORITY   = 5      # per priority point
_W_CONTEXT    = 0.001  # per context token
_W_FALLBACK   = -20    # penalty for fallback models


def score_model(task: Task, config: ModelConfig) -> float:
    """Score how well *config* fits *task*.  Higher is better.

    Returns a negative sentinel (``-1.0``) if the model lacks a
    **required** capability (hard filter).
    """
    required = _required_capabilities(task)
    model_caps = set(config.capabilities)

    # -- hard filter: at least one required capability must match --------
    if required and not required.intersection(model_caps):
        return -1.0

    score = 0.0

    # Capability match bonus
    score += len(required.intersection(model_caps)) * _W_CAPABILITY

    # Vision requirement — hard filter
    if task.requires_vision and ModelCapability.VISION not in model_caps:
        return -1.0

    # Priority bonus
    score += config.priority * _W_PRIORITY

    # Context length bonus (slight)
    score += config.context_length * _W_CONTEXT

    # Fallback penalty
    if config.is_fallback:
        score += _W_FALLBACK

    return score


def _required_capabilities(task: Task) -> set[ModelCapability]:
    """Derive required capabilities from the task."""
    caps: set[ModelCapability] = set()

    # From task type
    for cap in _TASK_CAPABILITIES.get(task.task_type, [ModelCapability.GENERAL]):
        caps.add(cap)

    # Vision adds VISION
    if task.requires_vision:
        caps.add(ModelCapability.VISION)

    return caps
