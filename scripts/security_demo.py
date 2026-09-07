#!/usr/bin/env python3
"""Deterministic CLI security demonstration for the AEGIS policy boundary."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from langchain_core.tools import StructuredTool

from routing.approval_note import ApprovalNote
from runtime.official_documents import OfficialDocumentWorkflow
from runtime.tool_policy import ApprovalRequired, PolicyDenied, PolicyEngine, ToolPolicy
from tools.registry import ToolRegistry


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="aegis-security-") as raw:
        root = Path(raw)
        (root / "allowed.txt").write_text("workspace evidence", encoding="utf-8")

        async def slow() -> str:
            await asyncio.sleep(0.2)
            return "finished"

        registry = ToolRegistry()
        registry.register(StructuredTool.from_function(lambda path: (root / Path(path)).read_text(encoding="utf-8"), name="read_file", description="read"))
        registry.register(StructuredTool.from_function(lambda url: "network", name="http_request", description="http"))
        registry.register(StructuredTool.from_function(lambda: "x" * 100, name="huge_output", description="output"))
        registry.register(StructuredTool.from_function(coroutine=slow, name="slow_tool", description="slow"))
        engine = PolicyEngine(root, deliverables_root=root / "deliverables",
                              approval_requester=lambda *_: False)
        engine.register(ToolPolicy("read_file", filesystem="workspace_only", timeout=2, max_output=1000))
        engine.register(ToolPolicy("http_request", filesystem="none", network=False))
        engine.register(ToolPolicy("huge_output", filesystem="none", max_output=16))
        engine.register(ToolPolicy("slow_tool", filesystem="none", timeout=0.05))
        engine.register(ToolPolicy("write_official_document", filesystem="deliverables_only", requires_approval=True))

        checks: dict[str, str] = {}
        checks["allowed_workspace_read"] = "ALLOW" if engine.evaluate("read_file", {"path": "allowed.txt"}).allowed else "FAIL"
        checks["passwd_denial"] = "DENIED" if not engine.evaluate("read_file", {"path": "/etc/passwd"}).allowed else "FAIL"
        checks["outside_write_denial"] = "DENIED" if not engine.evaluate("read_file", {"path": "../outside.txt"}).allowed else "FAIL"
        checks["network_denial"] = "DENIED" if not engine.evaluate("http_request", {"url": "https://example.com"}).allowed else "FAIL"

        async def execute(name: str, args: dict) -> object:
            try:
                return await engine.execute(registry, name, args)
            except ApprovalRequired:
                return "APPROVAL_REQUIRED"
            except asyncio.TimeoutError:
                return "TIMEOUT"
            except PolicyDenied:
                return "DENIED"

        async def run() -> None:
            allowed = await execute("read_file", {"path": "allowed.txt"})
            checks["allowed_workspace_read"] = "ALLOW" if allowed == "workspace evidence" else "FAIL"
            checks["timeout"] = await execute("slow_tool", {})
            result = await execute("huge_output", {})
            checks["huge_output"] = "TRUNCATED" if isinstance(result, str) and len(result) == 16 else "FAIL"
            checks["official_document"] = await execute("write_official_document", {"filename": "approval.docx"})

        asyncio.run(run())
        note = ApprovalNote(subject="Valve replacement", background="Inspection finding", proposal="Replace",
                            financial_implication="INR 1", dop_authority="Plant Head", recommendation="Approve")
        workflow = OfficialDocumentWorkflow(root / "deliverables")
        pending = workflow.generate(note, request_id="security-demo")
        checks["official_workflow_checkpoint"] = pending["status"].upper()
        print(json.dumps(checks, indent=2))
        return 0 if all(value not in {"FAIL", "DENIED"} or key.endswith("denial") for key, value in checks.items()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
