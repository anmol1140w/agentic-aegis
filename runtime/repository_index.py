"""Deterministic SQLite repository map and relevance index."""

from __future__ import annotations

import ast
import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any


_SKIP = {".git", ".venv", "__pycache__", "node_modules", "build", "dist", ".aegis"}
_CODE = {".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp"}


class RepositoryIndex:
    def __init__(self, root: str | Path, db_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.db_path = Path(db_path) if db_path else self.root / ".aegis" / "repository.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, size INTEGER, sha256 TEXT, language TEXT, is_test INTEGER);
            CREATE TABLE IF NOT EXISTS symbols(path TEXT, name TEXT, kind TEXT, line INTEGER);
            CREATE TABLE IF NOT EXISTS imports(path TEXT, target TEXT);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_files_test ON files(is_test);
        """)
        self.conn.commit()

    def refresh(self, *, max_files: int = 5000, max_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
        self.conn.execute("DELETE FROM symbols")
        self.conn.execute("DELETE FROM imports")
        seen = 0
        total_bytes = 0
        for path in self._files():
            if seen >= max_files or total_bytes >= max_bytes:
                break
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) > max_bytes - total_bytes:
                continue
            rel = str(path.relative_to(self.root))
            suffix = path.suffix.lower()
            language = suffix.lstrip(".") or "text"
            self.conn.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)",
                              (rel, len(data), hashlib.sha256(data).hexdigest(), language,
                               int(path.name.startswith("test_") or "tests" in path.parts)))
            if suffix in _CODE:
                self._extract_code(rel, data.decode("utf-8", errors="replace"), suffix)
            seen += 1
            total_bytes += len(data)
        self.conn.commit()
        return {"root": str(self.root), "file_count": seen, "bytes": total_bytes,
                "symbols": self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]}

    def summary(self) -> dict[str, Any]:
        rows = self.conn.execute("SELECT path, size, language, is_test FROM files ORDER BY path LIMIT 500").fetchall()
        return {"root": str(self.root), "file_count": self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                "files": [dict(row) for row in rows],
                "symbol_count": self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]}

    def search(self, query: str, *, limit: int = 30) -> list[dict[str, Any]]:
        terms = [term.lower() for term in re.findall(r"[A-Za-z0-9_./-]+", query) if len(term) > 1]
        if not terms:
            return []
        rows = self.conn.execute("SELECT path, size, language, is_test FROM files").fetchall()
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            path = str(row["path"]).lower()
            score = sum((4 if term in path else 0) for term in terms)
            symbol_hits = self.conn.execute("SELECT COUNT(*) FROM symbols WHERE path=? AND lower(name) LIKE ?",
                                             (row["path"], "%" + terms[0] + "%")).fetchone()[0]
            score += int(symbol_hits) * 3
            if score:
                scored.append((score, {**dict(row), "score": score}))
        return [item for _, item in sorted(scored, key=lambda pair: (-pair[0], pair[1]["path"]))[:limit]]

    def relevant_files(self, query: str, *, limit: int = 12) -> list[str]:
        return [item["path"] for item in self.search(query, limit=limit)]

    def _files(self):
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and not any(part in _SKIP for part in path.parts):
                yield path

    def _extract_code(self, rel: str, text: str, suffix: str) -> None:
        if suffix == ".py":
            try:
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.conn.execute("INSERT INTO symbols VALUES (?, ?, ?, ?)", (rel, node.name, type(node).__name__, node.lineno))
                    elif isinstance(node, ast.Import):
                        for item in node.names:
                            self.conn.execute("INSERT INTO imports VALUES (?, ?)", (rel, item.name))
                    elif isinstance(node, ast.ImportFrom):
                        self.conn.execute("INSERT INTO imports VALUES (?, ?)", (rel, node.module or ""))
            except SyntaxError:
                pass
        else:
            for match in re.finditer(r"\b(?:class|struct|function|func|fn|def)\s+([A-Za-z_][A-Za-z0-9_]*)", text):
                self.conn.execute("INSERT INTO symbols VALUES (?, ?, ?, ?)", (rel, match.group(1), "declaration", text.count("\n", 0, match.start()) + 1))
