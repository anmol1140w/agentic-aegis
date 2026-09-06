"""Structured local filesystem and OCR tools guarded by session permissions."""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool

from .ocr import PaddleOCRBackend
from .permissions import AccessMode, SessionPermissions

PermissionRequester = Callable[[AccessMode, Path, str], bool]
_MAX_FILE_BYTES = 10 * 1024 * 1024
_MAX_SEARCH_RESULTS = 1_000
_MAX_SEARCH_DEPTH = 12
_BINARY_DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}
_MAX_PDF_PAGES = 20


class FilesystemTools:
    """Safe local tools; external access is denied until approved for this session."""

    def __init__(self, permissions: SessionPermissions, requester: PermissionRequester | None = None) -> None:
        self.permissions = permissions
        self.requester = requester

    def _path(self, raw: str, mode: AccessMode, reason: str, *, directory_grant: bool = False) -> Path | dict[str, Any]:
        target = self.permissions.canonical(raw)
        if not self.permissions.allowed(target, mode):
            grant_target = target if directory_grant or target.is_dir() else target.parent
            approved = self.requester(mode, grant_target, reason) if self.requester else False
            if not approved:
                return self._error("PermissionDenied", f"{mode.value.upper()} access has not been granted for this path.", target)
            self.permissions.grant(grant_target, mode)
        return target

    @staticmethod
    def _error(error: str, message: str, path: Path | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"ok": False, "error": error, "message": message}
        if path is not None:
            result["path"] = str(path)
        return result

    @staticmethod
    def _item(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {"name": path.name, "path": str(path), "type": "directory" if path.is_dir() else "file",
                "size": stat.st_size if path.is_file() else None,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}

    def list_directory(self, path: str) -> dict[str, Any]:
        target = self._path(path, AccessMode.READ, "Need to inspect directory contents.", directory_grant=True)
        if isinstance(target, dict): return target
        if not target.is_dir(): return self._error("NotDirectory", "Path is not a directory.", target)
        try:
            return {"ok": True, "tool": "list_directory", "path": str(target),
                    "items": [self._item(item) for item in sorted(target.iterdir(), key=lambda p: p.name.lower())]}
        except OSError as exc:
            return self._error("FilesystemError", str(exc), target)

    def find_files(self, path: str, pattern: str = "*") -> dict[str, Any]:
        root = self._path(path, AccessMode.READ, "Need to search this directory tree.", directory_grant=True)
        if isinstance(root, dict): return root
        if not root.is_dir(): return self._error("NotDirectory", "Path is not a directory.", root)
        matches: list[dict[str, Any]] = []
        try:
            for item in root.rglob("*"):
                if len(matches) >= _MAX_SEARCH_RESULTS: break
                try: relative = item.relative_to(root)
                except ValueError: continue
                if len(relative.parts) > _MAX_SEARCH_DEPTH or not self.permissions.allowed(item, AccessMode.READ): continue
                if item.is_file() and fnmatch.fnmatch(item.name.lower(), pattern.lower()): matches.append(self._item(item))
        except OSError as exc:
            return self._error("FilesystemError", str(exc), root)
        return {"ok": True, "tool": "find_files", "path": str(root), "pattern": pattern,
                "items": matches, "truncated": len(matches) >= _MAX_SEARCH_RESULTS}

    def inspect_path(self, path: str) -> dict[str, Any]:
        target = self._path(path, AccessMode.READ, "Need to inspect this path.")
        if isinstance(target, dict): return target
        if not target.exists(): return self._error("NotFound", "Path does not exist.", target)
        return {"ok": True, "tool": "inspect_path", "item": self._item(target)}

    def read_file(self, path: str) -> dict[str, Any]:
        target = self._path(path, AccessMode.READ, "Need to read this file.")
        if isinstance(target, dict): return target
        if not target.is_file(): return self._error("NotFile", "Path is not a file.", target)
        if target.stat().st_size > _MAX_FILE_BYTES: return self._error("FileTooLarge", "File exceeds 10 MB read limit.", target)
        if target.suffix.lower() in _BINARY_DOCUMENT_EXTENSIONS:
            return self._error(
                "UnsupportedFileType",
                "This binary document format is not supported by read_file. Use a local document extractor before reading it.",
                target,
            )
        try: content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc: return self._error("FilesystemError", str(exc), target)
        return {"ok": True, "tool": "read_file", "path": str(target), "content": content}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._path(path, AccessMode.WRITE, "Need to create or modify this file.")
        if isinstance(target, dict): return target
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"ok": True, "tool": "write_file", "path": str(target), "bytes_written": len(content.encode())}
        except OSError as exc: return self._error("FilesystemError", str(exc), target)

    def edit_file(self, path: str, old_text: str, new_text: str) -> dict[str, Any]:
        target = self._path(path, AccessMode.WRITE, "Need to edit this file.")
        if isinstance(target, dict): return target
        if not target.is_file(): return self._error("NotFile", "Path is not a file.", target)
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            occurrences = content.count(old_text)
            if occurrences != 1:
                return self._error("EditMismatch", f"Expected one match, found {occurrences}.", target)
            target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
            return {"ok": True, "tool": "edit_file", "path": str(target)}
        except OSError as exc: return self._error("FilesystemError", str(exc), target)

    def extract_ocr(self, path: str) -> dict[str, Any]:
        target = self._path(path, AccessMode.READ, "Need to OCR this image.")
        if isinstance(target, dict): return target
        if target.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            return self._error(
                "UnsupportedFileType",
                "OCR supports only PNG, JPG, and JPEG images. Convert PDF pages to images before OCR.",
                target,
            )
        try:
            result = PaddleOCRBackend().extract(target)
            return {"ok": True, "tool": "extract_ocr", **result.model_dump(mode="json")}
        except Exception as exc: return self._error("OCRError", str(exc), target)

    def convert_pdf_to_images(self, path: str, output_dir: str | None = None) -> dict[str, Any]:
        source = self._path(path, AccessMode.READ, "Need to read this PDF for image conversion.")
        if isinstance(source, dict): return source
        if not source.is_file(): return self._error("NotFile", "Path is not a file.", source)
        if source.suffix.lower() != ".pdf":
            return self._error("UnsupportedFileType", "PDF to image conversion requires a .pdf file.", source)
        renderer = shutil.which("pdftoppm")
        if renderer is None:
            return self._error("DependencyUnavailable", "The local pdftoppm renderer is not installed.", source)

        destination = self.permissions.canonical(output_dir) if output_dir else source.parent / f"{source.stem}_pages"
        if not self.permissions.allowed(destination, AccessMode.WRITE):
            approved = self.requester(
                AccessMode.WRITE,
                destination,
                "Need to create rendered PDF page images for OCR.",
            ) if self.requester else False
            if not approved:
                return self._error("PermissionDenied", "WRITE access has not been granted for the output directory.", destination)
            self.permissions.grant(destination, AccessMode.WRITE)
        try:
            destination.mkdir(parents=True, exist_ok=True)
            prefix = destination / "page"
            subprocess.run(
                [renderer, "-png", "-r", "150", "-f", "1", "-l", str(_MAX_PDF_PAGES), str(source), str(prefix)],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            pages = sorted(destination.glob("page-*.png"))
            if not pages:
                return self._error("ConversionError", "No PDF pages were rendered.", source)
            return {
                "ok": True,
                "tool": "convert_pdf_to_images",
                "source": str(source),
                "output_dir": str(destination),
                "images": [str(page) for page in pages],
                "truncated": len(pages) >= _MAX_PDF_PAGES,
            }
        except subprocess.TimeoutExpired:
            return self._error("ConversionTimeout", "PDF conversion exceeded 120 seconds.", source)
        except (OSError, subprocess.CalledProcessError) as exc:
            message = getattr(exc, "stderr", None) or str(exc)
            return self._error("ConversionError", message.strip(), source)

    def ocr_pdf(self, path: str, output_dir: str | None = None) -> dict[str, Any]:
        """Convert a PDF to PNG pages, then run the existing OCR backend per page."""
        converted = self.convert_pdf_to_images(path, output_dir)
        if not converted.get("ok"):
            return converted
        backend = PaddleOCRBackend()
        pages: list[dict[str, Any]] = []
        for image_path in converted["images"]:
            try:
                pages.append(backend.extract(image_path).model_dump(mode="json"))
            except Exception as exc:
                pages.append({"ok": False, "path": image_path, "error": "OCRError", "message": str(exc)})
        return {
            "ok": all(page.get("error") is None for page in pages),
            "tool": "ocr_pdf",
            "source": converted["source"],
            "output_dir": converted["output_dir"],
            "pages": pages,
        }

    def as_langchain_tools(self) -> list[BaseTool]:
        @tool("list_directory")
        def list_directory(path: str) -> dict[str, Any]:
            """List a local directory after session read permission is granted."""
            return self.list_directory(path)
        @tool("find_files")
        def find_files(path: str, pattern: str = "*") -> dict[str, Any]:
            """Recursively find local files by glob pattern after read permission."""
            return self.find_files(path, pattern)
        @tool("inspect_path")
        def inspect_path(path: str) -> dict[str, Any]:
            """Inspect metadata for a local path after read permission."""
            return self.inspect_path(path)
        @tool("read_file")
        def read_file(path: str) -> dict[str, Any]:
            """Read a UTF-8 local text file after read permission."""
            return self.read_file(path)
        @tool("safe_read_file")
        def safe_read_file(path: str) -> dict[str, Any]:
            """Compatibility alias for permission-aware structured file reads."""
            return self.read_file(path)
        @tool("write_file")
        def write_file(path: str, content: str) -> dict[str, Any]:
            """Create or modify a text file after separate write permission."""
            return self.write_file(path, content)
        @tool("edit_file")
        def edit_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
            """Replace exactly one text occurrence after separate write permission."""
            return self.edit_file(path, old_text, new_text)
        @tool("extract_ocr")
        def extract_ocr(path: str) -> dict[str, Any]:
            """Run local PaddleOCR on a PNG/JPEG after read permission."""
            return self.extract_ocr(path)
        @tool("safe_extract_ocr")
        def safe_extract_ocr(path: str) -> dict[str, Any]:
            """Compatibility alias for the explicit OCR capability."""
            return self.extract_ocr(path)
        @tool("convert_pdf_to_images")
        def convert_pdf_to_images(path: str, output_dir: str | None = None) -> dict[str, Any]:
            """Render up to 20 PDF pages to local PNG files for explicit OCR."""
            return self.convert_pdf_to_images(path, output_dir)
        @tool("ocr_pdf")
        def ocr_pdf(path: str, output_dir: str | None = None) -> dict[str, Any]:
            """Convert a PDF to PNG pages and run local PaddleOCR on each page."""
            return self.ocr_pdf(path, output_dir)
        return [list_directory, find_files, inspect_path, read_file, safe_read_file, write_file, edit_file, extract_ocr, safe_extract_ocr, convert_pdf_to_images, ocr_pdf]
