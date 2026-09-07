"""Offline contextual routing with shadow mode and a promotion gate.

This module only chooses among already validated routing candidates. It has no
tool registry and cannot grant permissions, approve mutations, or change
filesystem/network policy. LinUCB uses a diagonal covariance approximation so
the router remains dependency-light and local.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class RoutingCandidate:
    name: str
    model: str
    capabilities: frozenset[str] = frozenset()
    quality: float = 1.0
    workflow: str = ""
    network: bool = False
    policy_compatible: bool = True
    security_validated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model,
                "capabilities": sorted(self.capabilities), "quality": self.quality,
                "workflow": self.workflow, "network": self.network,
                "policy_compatible": self.policy_compatible,
                "security_validated": self.security_validated}


@dataclass(frozen=True)
class BanditDecision:
    selected: RoutingCandidate
    baseline: RoutingCandidate
    eligible_candidates: tuple[RoutingCandidate, ...]
    mode: str
    fallback: bool = False
    reason: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.0
    confidence: float = 0.0
    exploration: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"selected": self.selected.to_dict(), "baseline": self.baseline.to_dict(),
                "eligible_candidates": [c.to_dict() for c in self.eligible_candidates],
                "mode": self.mode, "fallback": self.fallback, "reason": self.reason,
                "scores": self.scores, "uncertainty": self.uncertainty,
                "confidence": self.confidence, "exploration": self.exploration}


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"promoted": self.promoted, "reasons": list(self.reasons)}


class EvaluationGate:
    """Non-inferiority and security gate for adaptive routing promotion."""

    def __init__(self, minimum_accuracy: float = 0.95) -> None:
        self.minimum_accuracy = minimum_accuracy

    def compare(self, baseline: dict[str, Any], adaptive: dict[str, Any]) -> PromotionResult:
        reasons: list[str] = []
        if not bool(baseline.get("actual_workflow_execution", False)) or not bool(adaptive.get("actual_workflow_execution", False)):
            reasons.append("actual candidate workflow execution evidence is unavailable")
        if int(adaptive.get("executed_candidate_observations", 0)) <= 0:
            reasons.append("no executed adaptive candidate observations")
        if float(adaptive.get("task_success_rate", adaptive.get("accuracy", 0.0))) < float(baseline.get("task_success_rate", baseline.get("accuracy", 0.0))):
            reasons.append("task success below baseline")
        if float(adaptive.get("verification_pass_rate", 0.0)) < float(baseline.get("verification_pass_rate", 0.0)):
            reasons.append("verification pass rate below baseline")
        if int(adaptive.get("security_violations", 0)) != 0:
            reasons.append("security violations detected")
        if int(adaptive.get("unauthorized_tool_executions", 0)) != 0:
            reasons.append("unauthorized tool executions detected")
        if int(adaptive.get("approval_bypasses", 0)) != 0:
            reasons.append("approval bypass detected")
        if int(adaptive.get("critical_workflow_failures", 0)) != 0:
            reasons.append("critical workflow failure detected")
        if float(adaptive.get("schema_evidence_validity", 0.0)) < float(baseline.get("schema_evidence_validity", 0.0)):
            reasons.append("schema/evidence validity below baseline")
        if float(adaptive.get("average_latency_ms", float("inf"))) > float(baseline.get("average_latency_ms", 0.0)) * 1.10:
            reasons.append("average latency exceeds baseline by more than 10%")
        if float(adaptive.get("timeout_rate", 0.0)) > float(baseline.get("timeout_rate", 0.0)):
            reasons.append("timeout rate above baseline")
        if float(adaptive.get("adaptive_fallback_rate", 0.0)) > 0.15:
            reasons.append("adaptive fallback rate exceeds 15%")
        if float(adaptive.get("coverage", 0.0)) < 1.0:
            reasons.append("evaluation coverage is below 100%")
        return PromotionResult(not reasons, tuple(reasons))

    def open(self, report: dict[str, Any]) -> bool:
        """Compatibility API for the original single-report gate."""
        return float(report.get("accuracy", 0.0)) >= self.minimum_accuracy


class SQLiteBanditStore:
    """Durable local model state; no network or external service is used."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("CREATE TABLE IF NOT EXISTS bandit_models (name TEXT PRIMARY KEY, state_json TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS bandit_metadata (name TEXT PRIMARY KEY, state_json TEXT NOT NULL)")
        self.conn.commit()

    def load(self, name: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT state_json FROM bandit_models WHERE name=?", (name,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, name: str, state: dict[str, Any]) -> None:
        self.conn.execute("INSERT OR REPLACE INTO bandit_models VALUES (?, ?)", (name, json.dumps(state)))
        self.conn.commit()

    def load_metadata(self, name: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT state_json FROM bandit_metadata WHERE name=?", (name,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_metadata(self, name: str, state: dict[str, Any]) -> None:
        self.conn.execute("INSERT OR REPLACE INTO bandit_metadata VALUES (?, ?)", (name, json.dumps(state)))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class ContextualBanditRouter:
    """LinUCB router with deterministic shadow and controlled adaptive modes."""

    def __init__(self, baseline: Callable[..., Any] | None = None, *, gate: EvaluationGate | None = None,
                 alpha: float = 0.35, feature_count: int = 32,
                 store: SQLiteBanditStore | None = None, min_confidence: float = 0.05) -> None:
        self.policy_version = "linucb-diagonal-v1"
        self.router_name = "aegis-contextual-bandit"
        self.baseline = baseline
        self.gate = gate or EvaluationGate()
        self.alpha = max(0.0, alpha)
        self.feature_count = max(8, feature_count)
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self.store = store
        self.enabled = False
        self.mode = "shadow"
        self._arms: dict[str, dict[str, list[float]]] = {}
        self._observations = 0
        self._contexts: set[str] = set()
        self._action_counts: dict[str, int] = {}
        self._context_action_counts: dict[str, dict[str, int]] = {}
        self._exploration_count = 0
        self._exploitation_count = 0
        self._last_decisions: list[dict[str, Any]] = []
        self._evaluation_observations = 0
        self._outcome_stats: dict[str, dict[str, float]] = {}
        self._invalid_actions: dict[str, set[str]] = {}
        if self.store:
            metadata = self.store.load_metadata(self.router_name) or {}
            self._observations = int(metadata.get("observations", 0))
            self._contexts = set(metadata.get("contexts", []))
            self._action_counts = {str(k): int(v) for k, v in metadata.get("action_counts", {}).items()}
            self._context_action_counts = {
                str(context): {str(action): int(count) for action, count in actions.items()}
                for context, actions in metadata.get("context_action_counts", {}).items()
            }
            self._exploration_count = int(metadata.get("exploration_count", 0))
            self._exploitation_count = int(metadata.get("exploitation_count", 0))
            self._evaluation_observations = int(metadata.get("evaluation_observations", 0))
            self._outcome_stats = {
                str(action): {str(key): float(value) for key, value in stats.items()}
                for action, stats in metadata.get("outcome_stats", {}).items()
            }
            self._invalid_actions = {
                str(context): set(map(str, actions))
                for context, actions in metadata.get("invalid_actions", {}).items()
            }
            for action in self._action_counts:
                self._arm(action)

    def enable_after_evaluation(self, report: dict[str, Any], *, baseline: dict[str, Any] | None = None) -> None:
        if baseline is not None:
            result = self.gate.compare(baseline, report)
            if not result.promoted:
                raise RuntimeError("RL routing gate is closed: " + "; ".join(result.reasons))
        elif not self.gate.open(report):
            raise RuntimeError("RL routing gate is closed: golden-task accuracy is below threshold")
        self.enabled = True
        self.mode = "adaptive"

    def disable(self) -> None:
        self.enabled = False
        self.mode = "shadow"

    def _features(self, context: dict[str, Any]) -> list[float]:
        values: list[str] = []
        for key in sorted(context):
            value = context[key]
            if isinstance(value, (list, tuple, set)):
                values.extend(f"{key}={item}" for item in sorted(map(str, value)))
            else:
                values.append(f"{key}={value}")
        vector = [0.0] * self.feature_count
        for value in values:
            digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.feature_count
            vector[index] += 1.0 if digest[0] & 1 else -1.0
        vector[0] = 1.0
        return vector

    def _arm(self, name: str) -> dict[str, list[float]]:
        if name not in self._arms:
            loaded = self.store.load(name) if self.store else None
            self._arms[name] = loaded or {"a": [1.0] * self.feature_count, "b": [0.0] * self.feature_count, "n": 0}
            self._arms[name].setdefault("n", 0)
        return self._arms[name]

    def _score(self, candidate: RoutingCandidate, vector: list[float]) -> tuple[float, float]:
        arm = self._arm(candidate.name)
        theta = [b / max(a, 1e-9) for a, b in zip(arm["a"], arm["b"])]
        mean = sum(t * x for t, x in zip(theta, vector))
        uncertainty = math.sqrt(sum((x * x) / max(a, 1e-9) for a, x in zip(arm["a"], vector)))
        return mean + self.alpha * uncertainty, uncertainty

    @staticmethod
    def _context_id(context: dict[str, Any]) -> str:
        canonical = json.dumps(context, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def select(self, context: dict[str, Any], candidates: Iterable[RoutingCandidate], baseline: RoutingCandidate,
               *, mode: str | None = None) -> BanditDecision:
        context_id = self._context_id(context)
        invalid = self._invalid_actions.get(context_id, set())
        eligible = tuple(c for c in candidates
                         if c.quality >= float(context.get("quality_required", 0.0))
                         and c.security_validated and c.policy_compatible and not c.network
                         and c.name not in invalid)
        if not eligible or baseline not in eligible:
            return BanditDecision(baseline, baseline, eligible, "fallback", True, "invalid or incompatible candidate set")
        requested = mode or self.mode
        vector = self._features(context)
        scored = {candidate.name: self._score(candidate, vector) for candidate in eligible}
        scores = {name: score for name, (score, _) in scored.items()}
        uncertainties = {name: uncertainty for name, (_, uncertainty) in scored.items()}
        learned = max(eligible, key=lambda candidate: (scores[candidate.name], candidate.name))
        ordered = sorted(scores.values(), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        uncertainty = uncertainties[learned.name]
        selected_count = int(self._arm(learned.name).get("n", 0))
        observed_counts = [int(self._arm(candidate.name).get("n", 0)) for candidate in eligible]
        exploration = selected_count == 0 or selected_count < max(observed_counts)
        confidence = max(0.0, min(1.0, margin / (1.0 + uncertainty)))
        self._contexts.add(self._context_id(context))
        self._last_decisions.append({"timestamp": time.time(), "context_id": self._context_id(context),
                                     "selected": learned.name, "mode": requested,
                                     "exploration": exploration, "confidence": confidence,
                                     "uncertainty": uncertainty})
        self._last_decisions = self._last_decisions[-100:]
        if requested != "adaptive" or not self.enabled:
            return BanditDecision(baseline, baseline, eligible, "shadow", False,
                                  "deterministic router remains authoritative", scores, uncertainty,
                                  confidence, exploration)
        if confidence < self.min_confidence:
            return BanditDecision(baseline, baseline, eligible, "adaptive", True,
                                  "low-confidence adaptive decision; deterministic fallback", scores,
                                  uncertainty, confidence, exploration)
        return BanditDecision(learned, baseline, eligible, "adaptive", False,
                              "selected by LinUCB" if learned.name != baseline.name else "bandit agreed with baseline", scores,
                              uncertainty, confidence, exploration)

    def record_outcome(self, decision: BanditDecision, context: dict[str, Any], reward: float,
                       *, latency_ms: float | None = None, success: bool | None = None,
                       verification: bool | None = None, scope: str = "training",
                       invalid: bool = False) -> None:
        """Update only the arm that actually ran; shadow predictions are never treated as outcomes."""
        arm_name = decision.selected.name if decision.mode == "adaptive" else decision.baseline.name
        if invalid:
            context_id = self._context_id(context)
            self._invalid_actions.setdefault(context_id, set()).add(arm_name)
            if self.store:
                self.store.save_metadata(self.router_name, {
                    "policy_version": self.policy_version, "observations": self._observations,
                    "contexts": sorted(self._contexts), "action_counts": self._action_counts,
                    "context_action_counts": self._context_action_counts,
                    "exploration_count": self._exploration_count,
                    "exploitation_count": self._exploitation_count,
                    "evaluation_observations": self._evaluation_observations,
                    "outcome_stats": self._outcome_stats,
                    "invalid_actions": {key: sorted(value) for key, value in self._invalid_actions.items()},
                })
            return
        if scope == "evaluation":
            self.record_evaluation_outcome(arm_name, reward, latency_ms=latency_ms,
                                           success=success, verification=verification)
            return
        arm = self._arm(arm_name)
        vector = self._features(context)
        for index, value in enumerate(vector):
            arm["a"][index] += value * value
            arm["b"][index] += value * float(reward)
        arm["n"] = int(arm.get("n", 0)) + 1
        self._observations += 1
        self._contexts.add(self._context_id(context))
        self._action_counts[arm_name] = self._action_counts.get(arm_name, 0) + 1
        stats = self._outcome_stats.setdefault(arm_name, {"observations": 0.0, "reward_sum": 0.0,
                                                            "latency_sum_ms": 0.0, "successes": 0.0,
                                                            "verifications": 0.0})
        stats["observations"] += 1
        stats["reward_sum"] += float(reward)
        if latency_ms is not None:
            stats["latency_sum_ms"] += max(0.0, float(latency_ms))
        stats["successes"] += float(bool(success))
        stats["verifications"] += float(bool(verification))
        context_label = str(context.get("task_type", context.get("workflow", "unknown")))
        per_context = self._context_action_counts.setdefault(context_label, {})
        per_context[arm_name] = per_context.get(arm_name, 0) + 1
        if decision.exploration:
            self._exploration_count += 1
        else:
            self._exploitation_count += 1
        if self.store:
            self.store.save(arm_name, arm)
            self.store.save_metadata(self.router_name, {
                "policy_version": self.policy_version, "observations": self._observations,
                "contexts": sorted(self._contexts), "action_counts": self._action_counts,
                "context_action_counts": self._context_action_counts,
                "exploration_count": self._exploration_count,
                "exploitation_count": self._exploitation_count,
                "evaluation_observations": self._evaluation_observations,
                "outcome_stats": self._outcome_stats,
                "invalid_actions": {key: sorted(value) for key, value in self._invalid_actions.items()},
            })

    def record_evaluation_outcome(self, action: str, reward: float, *, latency_ms: float | None = None,
                                  success: bool | None = None, verification: bool | None = None) -> None:
        """Record an executed evaluation result without training the policy."""
        self._evaluation_observations += 1
        stats = self._outcome_stats.setdefault(str(action), {"observations": 0.0, "reward_sum": 0.0,
                                                              "latency_sum_ms": 0.0, "successes": 0.0,
                                                              "verifications": 0.0})
        stats["observations"] += 1
        stats["reward_sum"] += float(reward)
        if latency_ms is not None:
            stats["latency_sum_ms"] += max(0.0, float(latency_ms))
        stats["successes"] += float(bool(success))
        stats["verifications"] += float(bool(verification))
        if self.store:
            self.store.save_metadata(self.router_name, {
                "policy_version": self.policy_version, "observations": self._observations,
                "contexts": sorted(self._contexts), "action_counts": self._action_counts,
                "context_action_counts": self._context_action_counts,
                "exploration_count": self._exploration_count,
                "exploitation_count": self._exploitation_count,
                "evaluation_observations": self._evaluation_observations,
                "outcome_stats": self._outcome_stats,
            })

    def diagnostics(self) -> dict[str, Any]:
        arms: dict[str, Any] = {}
        for name, arm in self._arms.items():
            arms[name] = {"observations": int(arm.get("n", 0)),
                          "a": list(arm.get("a", [])), "b": list(arm.get("b", [])),
                          "theta": [b / max(a, 1e-9) for a, b in zip(arm.get("a", []), arm.get("b", []))]}
        total_decisions = self._exploration_count + self._exploitation_count
        return {"router": self.router_name, "policy_version": self.policy_version,
                "algorithm": "LinUCB (diagonal covariance)", "alpha": self.alpha,
                "feature_count": self.feature_count, "training_observations": self._observations,
                "min_confidence": self.min_confidence,
                "unique_contexts": len(self._contexts), "actions": sorted(self._action_counts),
                "action_selection_frequencies": dict(self._action_counts),
                "action_selection_by_task_type": self._context_action_counts,
                "exploration_count": self._exploration_count,
                "exploitation_count": self._exploitation_count,
                "exploration_rate": self._exploration_count / max(1, total_decisions),
                "evaluation_observations": self._evaluation_observations,
                "outcome_stats": {
                    action: {**stats,
                             "average_reward": stats["reward_sum"] / max(1.0, stats["observations"]),
                             "average_latency_ms": stats["latency_sum_ms"] / max(1.0, stats["observations"]),
                             "success_rate": stats["successes"] / max(1.0, stats["observations"]),
                             "verification_rate": stats["verifications"] / max(1.0, stats["observations"])}
                    for action, stats in self._outcome_stats.items()
                },
                "invalid_actions": {key: sorted(value) for key, value in self._invalid_actions.items()},
                "arms": arms, "recent_decisions": list(self._last_decisions),
                "enabled": self.enabled, "mode": self.mode}

    def record_observed_outcome(self, decision: dict[str, Any], context: dict[str, Any], reward: float,
                                *, latency_ms: float | None = None, success: bool | None = None,
                                verification: bool | None = None, invalid: bool = False) -> None:
        """Record an outcome from serialized graph telemetry without inventing arms."""
        selected = decision.get("selected") or {}
        baseline = decision.get("baseline") or selected
        def candidate(item: dict[str, Any]) -> RoutingCandidate:
            return RoutingCandidate(
                name=str(item.get("name")), model=str(item.get("model", "")),
                capabilities=frozenset(item.get("capabilities", [])),
                quality=float(item.get("quality", 1.0)), workflow=str(item.get("workflow", "")),
                network=bool(item.get("network", False)),
                policy_compatible=bool(item.get("policy_compatible", True)),
                security_validated=bool(item.get("security_validated", True)),
            )
        candidates = tuple(candidate(item) for item in decision.get("eligible_candidates", []))
        if not selected.get("name") or not baseline.get("name"):
            return
        observed = BanditDecision(
            selected=candidate(selected),
            baseline=candidate(baseline),
            eligible_candidates=candidates, mode=str(decision.get("mode", "shadow")),
            uncertainty=float(decision.get("uncertainty", 0.0)),
            confidence=float(decision.get("confidence", 0.0)),
            exploration=bool(decision.get("exploration", False)),
        )
        self.record_outcome(observed, context, reward, latency_ms=latency_ms,
                            success=success, verification=verification, invalid=invalid)

    def route(self, *args: Any, **kwargs: Any) -> Any:
        if "context" in kwargs and "candidates" in kwargs and "baseline" in kwargs:
            return self.select(kwargs["context"], kwargs["candidates"], kwargs["baseline"], mode=kwargs.get("mode"))
        if self.baseline is None:
            raise TypeError("route requires context/candidates/baseline or a baseline callable")
        return self.baseline(*args, **kwargs)
