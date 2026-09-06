"""Tests for session-scoped local filesystem tools."""

from __future__ import annotations

from pathlib import Path

from tools.filesystem import FilesystemTools
from tools.permissions import AccessMode, SessionPermissions


def test_external_read_requires_then_uses_session_grant(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "notes.txt").write_text("pump-101")
    prompts: list[tuple[AccessMode, Path]] = []

    def approve(mode: AccessMode, path: Path, _reason: str) -> bool:
        prompts.append((mode, path))
        return True

    tools = FilesystemTools(SessionPermissions(), approve)
    result = tools.read_file(str(external / "notes.txt"))
    assert result["ok"] is True and result["content"] == "pump-101"
    assert prompts == [(AccessMode.READ, external)]
    assert tools.read_file(str(external / "notes.txt"))["ok"] is True
    assert len(prompts) == 1


def test_denied_external_path_is_not_read(tmp_path: Path) -> None:
    path = tmp_path / "private.txt"
    path.write_text("private")
    tools = FilesystemTools(SessionPermissions(), lambda *_: False)
    result = tools.read_file(str(path))
    assert result["ok"] is False
    assert result["error"] == "PermissionDenied"


def test_write_is_separate_from_read_permission(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    prompts: list[AccessMode] = []
    tools = FilesystemTools(SessionPermissions(), lambda mode, *_: prompts.append(mode) or True)
    assert tools.list_directory(str(external))["ok"] is True
    assert tools.write_file(str(external / "result.txt"), "done")["ok"] is True
    assert prompts == [AccessMode.READ, AccessMode.WRITE]
    assert (external / "result.txt").read_text() == "done"


def test_recursive_search_and_traversal_canonicalization(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "drawings").mkdir(parents=True)
    (root / "drawings" / "pump-101.jpg").write_bytes(b"image")
    tools = FilesystemTools(SessionPermissions([root]))
    found = tools.find_files(str(root / "drawings" / ".."), "*pump*")
    assert found["ok"] is True
    assert [item["name"] for item in found["items"]] == ["pump-101.jpg"]


def test_ocr_is_not_automatic_and_requires_read_permission(tmp_path: Path) -> None:
    image = tmp_path / "drawing.jpg"
    image.write_bytes(b"not an image")
    tools = FilesystemTools(SessionPermissions(), lambda *_: False)
    result = tools.extract_ocr(str(image))
    assert result["ok"] is False
    assert result["error"] == "PermissionDenied"


def test_session_permissions_do_not_persist_between_instances(tmp_path: Path) -> None:
    path = tmp_path / "external"
    path.mkdir()
    first = SessionPermissions()
    first.grant(path, AccessMode.READ)
    assert first.allowed(path, AccessMode.READ) is True
    assert SessionPermissions().allowed(path, AccessMode.READ) is False
