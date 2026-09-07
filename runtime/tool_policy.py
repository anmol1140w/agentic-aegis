"""Central policy gateway between agents and tools.

Tool implementations still own their local validation, but every invocation
must pass this gateway first.  The gateway is deliberately deny-by-default for
unknown tools and keeps approval, scope, budgets, and audit decisions in one
serializable object.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tools.registry import ToolMeta, ToolRegistry


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    allowed: bool = True
    network: bool = False
    filesystem: str = "workspace_only"  # workspace_only|deliverables_only|temp_only|none
    requires_approval: bool = False
    timeout: float = 30.0
    max_output: int = 20_000
    destructive: bool = False
    external_side_effect: bool = False


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    approval_required: bool = False
    reason: str = ""
    policy: ToolPolicy | None = None
    paths: tuple[str, ...] = ()


class PolicyDenied(PermissionError):
    """Raised when a tool violates the central policy."""


class ApprovalRequired(PolicyDenied):
    """Raised when a human checkpoint must be resolved before execution."""

    def __init__(self, request_id: str, message: str):
        super().__init__(message)
        self.request_id = request_id


# Tool implementations still contain legacy approval callbacks.  The policy
# gateway is the canonical checkpoint, so it marks an invocation as approved
# while calling the legacy implementation; those callbacks can then observe
# the grant without prompting a second time.
_APPROVAL_GRANTED = contextvars.ContextVar("aegis_policy_approval_granted", default=False)


def policy_approval_granted() -> bool:
    return bool(_APPROVAL_GRANTED.get())


class PolicyEngine:
    # TODO: normalize legacy LangChain tool argument schemas through an
    # adapter into one canonical ToolRequest before policy evaluation.
    # Compatibility aliases (for example read_file(file_path) vs
    # workspace.read_file(path)) must remain supported during migration.
    def __init__(self, workspace_root: str | Path, *, deliverables_root: str | Path | None = None,
                 temporary_root: str | Path | None = None,
                 approval_requester: Callable[[str, str, str], bool] | None = None,
                 audit: Callable[..., Any] | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.deliverables_root = Path(deliverables_root or self.workspace_root / "outputs").resolve()
        self.temporary_root = Path(temporary_root or self.workspace_root / "temporary").resolve()
        self.approval_requester = approval_requester
        self.audit = audit
        self._policies: dict[str, ToolPolicy] = {}

    def register(self, policy: ToolPolicy) -> None:
        self._policies[policy.name] = policy

    def register_from_meta(self, meta: ToolMeta) -> None:
        permissions = meta.permissions
        self.register(ToolPolicy(
            name=meta.name,
            network=bool(permissions.get("network")),
            filesystem="workspace_only" if meta.workspace_scoped else "none",
            requires_approval=meta.requires_approval or bool(permissions.get("write"))
                         or bool(permissions.get("delete")) or bool(permissions.get("external_side_effect")),
            timeout=float(meta.timeout_seconds),
            max_output=meta.max_output_bytes,
            destructive=bool(permissions.get("delete")),
            external_side_effect=bool(permissions.get("external_side_effect")),
        ))

    def policy_for(self, name: str) -> ToolPolicy | None:
        return self._policies.get(name)

    def evaluate(self, name: str, args: dict[str, Any] | None = None) -> PolicyDecision:
        policy = self._policies.get(name)
        if policy is None:
            return PolicyDecision(False, reason=f"Tool '{name}' has no registered policy")
        if not policy.allowed:
            return PolicyDecision(False, reason=f"Tool '{name}' is denied by policy", policy=policy)
        args = args or {}
        if not policy.network and any(key in args for key in ("url", "uri", "host", "endpoint")):
            return PolicyDecision(False, reason=f"Network access is disabled for tool '{name}'", policy=policy)
        paths = tuple(self._paths(args))
        for raw in paths:
            if not self._path_allowed(raw, policy.filesystem):
                return PolicyDecision(False, reason=f"Path '{raw}' is outside the {policy.filesystem} scope", policy=policy, paths=paths)
        return PolicyDecision(True, approval_required=policy.requires_approval,
                              reason="approval required" if policy.requires_approval else "allowed",
                              policy=policy, paths=paths)

    async def execute(self, registry: ToolRegistry, name: str, args: dict[str, Any] | None = None,
                      *, task_id: str = "", approve: Callable[[str, str, str], bool] | None = None,
                      tool_override: Any | None = None) -> Any:
        args = args or {}
        decision = self.evaluate(name, args)
        self._record("tool_policy", name=name, task_id=task_id, decision=decision.reason, allowed=decision.allowed)
        if not decision.allowed:
            raise PolicyDenied(decision.reason)
        approval_token = None
        if decision.approval_required:
            requester = approve or self.approval_requester
            request_id = f"approval:{task_id or 'run'}:{name}"
            if requester is None or not requester(request_id, name, self._approval_details(args)):
                raise ApprovalRequired(request_id, f"Human approval required for tool '{name}'")
            approval_token = _APPROVAL_GRANTED.set(True)
        tool = tool_override or registry.get(name)
        started = time.monotonic()
        try:
            result = tool.ainvoke(args)
            value = await asyncio.wait_for(result, timeout=decision.policy.timeout) if inspect.isawaitable(result) else result
            # Preserve structured tool results for the graph and verification
            # layers.  Only textual payloads are truncated here; converting a
            # dict result to text would silently destroy evidence fields such
            # as ``ok``, ``content`` and ``exit_code``.
            if isinstance(value, str):
                return value[:decision.policy.max_output]
            return value
        finally:
            if approval_token is not None:
                _APPROVAL_GRANTED.reset(approval_token)
            self._record("tool_execution", name=name, task_id=task_id,
                         duration_ms=round((time.monotonic() - started) * 1000, 2))

    def wrap_callables(self, tools: dict[str, Callable[..., Any]], *, task_id: str = "") -> dict[str, Callable[..., Any]]:
        """Return compatibility adapters for specialist callables.

        Specialists historically receive plain Python functions. These adapters
        preserve that synchronous contract while applying the same policy
        decision before every invocation. Async-native callers should use
        :meth:`execute` directly.
        """
        wrapped: dict[str, Callable[..., Any]] = {}
        for name, function in tools.items():
            if self.policy_for(name) is None:
                self.register(ToolPolicy(name=name, filesystem="workspace_only",
                                         requires_approval=name in {"edit_file", "create_file", "create_python_script", "execute_command", "restore_checkpoint"},
                                         destructive=name == "restore_checkpoint"))
            wrapped[name] = functools.wraps(function)(self._wrap_callable(name, function, task_id=task_id))
        return wrapped

    def _wrap_callable(self, name: str, function: Callable[..., Any], *, task_id: str) -> Callable[..., Any]:
        def invoke(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = inspect.signature(function).bind_partial(*args, **kwargs)
                parameters = dict(bound.arguments)
            except (TypeError, ValueError):
                parameters = dict(kwargs)
            decision = self.evaluate(name, parameters)
            self._record("tool_policy", name=name, task_id=task_id, decision=decision.reason, allowed=decision.allowed)
            if not decision.allowed:
                raise PolicyDenied(decision.reason)
            if decision.approval_required:
                requester = self.approval_requester
                request_id = f"approval:{task_id or 'run'}:{name}"
                if requester is None or not requester(request_id, name, self._approval_details(parameters)):
                    raise ApprovalRequired(request_id, f"Human approval required for tool '{name}'")
            started = time.monotonic()
            value = function(*args, **kwargs)
            if inspect.isawaitable(value):
                raise PolicyDenied("Async callable must be invoked through the async policy gateway")
            policy = decision.policy
            if isinstance(value, str) and policy:
                value = value[:policy.max_output]
            self._record("tool_execution", name=name, task_id=task_id,
                         duration_ms=round((time.monotonic() - started) * 1000, 2))
            return value
        return invoke

    def _path_allowed(self, raw: str, scope: str) -> bool:
        if scope == "none":
            return True
        path = Path(raw)
        resolved = path.resolve() if path.is_absolute() else (self.workspace_root / path).resolve()
        root = {"workspace_only": self.workspace_root, "deliverables_only": self.deliverables_root,
                "temp_only": self.temporary_root}.get(scope)
        return root is not None and resolved == root or (root is not None and resolved.is_relative_to(root))

    @staticmethod
    def _paths(args: dict[str, Any]) -> list[str]:
        keys = {"path", "file_path", "filepath", "file", "directory", "output_path", "input_path"}
        return [str(value) for key, value in args.items() if key in keys and isinstance(value, (str, Path))]

    @staticmethod
    def _approval_details(args: dict[str, Any]) -> str:
        return "Arguments: " + ", ".join(f"{k}={str(v)[:160]}" for k, v in args.items())

    def _record(self, event: str, **data: Any) -> None:
        if self.audit:
            self.audit(event, metadata=data)


def build_policy_engine(registry: ToolRegistry, workspace_root: str | Path, **kwargs: Any) -> PolicyEngine:
    engine = PolicyEngine(workspace_root, **kwargs)
    for name in registry.list_names():
        engine.register_from_meta(registry.get_meta(name))
    # Reserve the high-value gates now so future integrations cannot silently
    # become side-effecting tools without an explicit policy entry.
    engine.register(ToolPolicy("write_official_document", filesystem="deliverables_only",
                              requires_approval=True, destructive=False))
    return engine
