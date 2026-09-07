"""Golden-task evaluation and pre-RL gating."""

from .golden import evaluate_golden_tasks, load_golden_tasks
from .routing import evaluate_baseline_vs_adaptive, reward_for_outcome

__all__ = ["evaluate_golden_tasks", "load_golden_tasks", "evaluate_baseline_vs_adaptive", "reward_for_outcome"]
