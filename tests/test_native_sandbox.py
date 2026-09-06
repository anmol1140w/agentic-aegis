from pathlib import Path

from tools.bwrap_sandbox import BubblewrapSandbox
from tools.native_sandbox import NativeSandbox
from tools.sandbox_config import SandboxConfig


def test_configured_bwrap_dry_run_is_explicit_and_network_off(tmp_path: Path) -> None:
    config = SandboxConfig(memory_limit_mb=128, max_pids=32, timeout_seconds=10)
    sandbox = BubblewrapSandbox(tmp_path, config)
    result = sandbox.dry_run("python -c 'print(1)'", tmp_path)
    assert result["ok"] is True
    assert result["network"] == "none"
    assert "--clearenv" in result["argv"]
    assert "--cap-drop" in result["argv"]
    assert result["resource_limits"]["max_pids"] == 32


def test_native_sandbox_persists_transfers_and_cleans_up(tmp_path: Path) -> None:
    with NativeSandbox(root=tmp_path) as sandbox:
        assert sandbox.upload_files([("src/input.txt", b"hello")])[0]["success"] is True
        downloaded = sandbox.download_files(["src/input.txt"])[0]
        assert downloaded["content"] == b"hello"
        assert sandbox.list_files() == ["src/input.txt"]
        assert sandbox.dry_run("printf hello")["ok"] is True


def test_native_sandbox_rejects_path_escape(tmp_path: Path) -> None:
    with NativeSandbox(root=tmp_path) as sandbox:
        result = sandbox.upload_files([("../outside.txt", b"no")])[0]
        assert result["success"] is False
