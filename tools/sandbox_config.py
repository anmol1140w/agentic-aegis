"""Configuration for AEGIS native Bubblewrap sandboxes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SandboxConfig:
    """Least-privilege defaults shared by command and native sandbox APIs."""

    memory_limit_mb: int = 512
    max_pids: int = 256
    cpu_weight: int = 100
    timeout_seconds: float = 60.0
    max_output_bytes: int = 256 * 1024
    network_access: bool = False
    gpu: bool = False
    bwrap_path: str = "bwrap"
    extra_ro_binds: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    extra_env: dict[str, str] = field(default_factory=dict)
    env_allow: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.memory_limit_mb < 1 or self.max_pids < 1 or not 1 <= self.cpu_weight <= 10_000:
            raise ValueError("invalid sandbox resource limits")
        if self.timeout_seconds <= 0 or self.max_output_bytes < 1:
            raise ValueError("timeout and output limits must be positive")
