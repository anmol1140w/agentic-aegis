from pathlib import Path

import pytest

from tools.cgroup_limits import CgroupLimitError, CgroupV2Limit, cgroup_mode


def test_cgroup_limit_configures_and_cleans_up_fake_v2_tree(tmp_path: Path) -> None:
    (tmp_path / "cgroup.controllers").write_text("memory pids cpu", encoding="ascii")
    limit = CgroupV2Limit(root=tmp_path, memory_mb=64, max_pids=8, cpu_weight=200)
    limit.create()
    assert (limit.path / "memory.max").read_text() == str(64 * 1024 * 1024)
    assert (limit.path / "pids.max").read_text() == "8"
    assert (limit.path / "cpu.weight").read_text() == "200"
    limit.attach(1234)
    assert (limit.path / "cgroup.procs").read_text() == "1234"
    limit.destroy()
    assert not limit.path.exists()


def test_cgroup_limit_requires_v2(tmp_path: Path) -> None:
    with pytest.raises(CgroupLimitError):
        CgroupV2Limit(root=tmp_path).create()


def test_cgroup_mode_defaults_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEGIS_CGROUP_MODE", raising=False)
    assert cgroup_mode() == "auto"
    monkeypatch.setenv("AEGIS_CGROUP_MODE", "invalid")
    assert cgroup_mode() == "auto"
