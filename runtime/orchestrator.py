"""LangGraph orchestrator — the primary agent runtime.

Builds a ``StateGraph`` with nodes:

    START → classify → route ┬→ execute_tool ──→ END (direct deterministic tool)
                             └→ execute_model ─→ END (LLM reasoning / vision)

The orchestrator guarantees:
1. Deterministic direct tool tasks (calculator, file ops) bypass the LLM completely.
2. Vision tasks pass base64 image data directly to the vision-capable model.
3. Serialization-safe state (primitive dicts/lists) eliminates checkpoint warnings.
4. Latency is measured per phase (classification, routing, tool, model, total).
"""

from __future__ import annotations

import json
import time
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from models.registry import ModelRegistry
from runtime.state import AgentState, make_trace_dict
from runtime.agents import MasterAgent, build_default_agent_registry
from runtime.coding_graph import run_coding_graph
from runtime.task_graph import run_task_graph
from runtime.approvals import ApprovalManager
from storage.outputs import OutputStore
from storage.graph_state import SQLiteGraphStateStore
from storage.telemetry import SQLiteTelemetryStore
from security.audit import AuditLogger
from security.network import NetworkMonitor
from tools.calculator import calculator
from tools.files import create_directory, list_files, read_file, write_file
from tools.ocr import extract_ocr
from tools.filesystem import FilesystemTools
from tools.workspace import WorkspaceReadTools
from tools.documents import DocumentTools
from tools.vision import VisionRuntime
from tools.permissions import AccessMode, SessionPermissions
from tools.registry import ToolRegistry
from tools.mcp_adapter import AegisMCPAdapter
from runtime.prompts import SYSTEM_PROMPT, TOOL_LOOP_PROMPT
from runtime.tool_policy import PolicyDenied, build_policy_engine
from runtime.official_documents import OfficialDocumentWorkflow
from routing.approval_note import ApprovalNote
from routing.adaptive_router import ContextualBanditRouter, SQLiteBanditStore
from evaluation.routing import reward_for_outcome

_SYSTEM_PROMPT = SYSTEM_PROMPT
_TOOL_LOOP_PROMPT = TOOL_LOOP_PROMPT


class Orchestrator:
    """LangGraph-based agent orchestrator."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        audit: AuditLogger | None = None,
        network: NetworkMonitor | None = None,
        availability: dict[str, bool] | None = None,
        output_store: OutputStore | None = None,
        approvals: ApprovalManager | None = None,
        max_iterations: int = 20,
        workspace_root: str | Path | None = None,
    ):
        self.registry = registry
        # Deprecated compatibility graph dependencies are loaded lazily.  The
        # production CLI/API path never constructs or invokes this graph;
        # run_master enters runtime.task_graph instead.
        from routing.classifier import TaskClassifier
        from routing.router import ModelRouter
        self.classifier = TaskClassifier()
        self.router = ModelRouter(registry)
        self.audit = audit or AuditLogger()
        self.network = network or NetworkMonitor()
        self.availability = availability
        self.output_store = output_store or OutputStore()
        self.workspace_root = Path(workspace_root or self.output_store.workspace_dir).resolve()
        self.graph_state_store = SQLiteGraphStateStore(self.workspace_root / ".aegis" / "runs.db")
        self.telemetry_store = SQLiteTelemetryStore(self.workspace_root / ".aegis" / "runs.db")
        self.adaptive_router = ContextualBanditRouter(
            store=SQLiteBanditStore(self.workspace_root / ".aegis" / "runs.db")
        )
        self.routing_mode = os.getenv("AEGIS_ROUTING_MODE", "shadow").strip().lower()
        if self.routing_mode not in {"shadow", "adaptive"}:
            self.routing_mode = "shadow"
        self.official_documents = OfficialDocumentWorkflow(
            self.output_store.artifacts_dir,
        )
        self.approvals = approvals or ApprovalManager()
        self.max_iterations = min(20, max(1, max_iterations))
        self._session_task_type: dict[str, str] = {}   # NEW: thread_id -> last task_type
        self._approval_fn = None  # injectable for tests; None → interactive CLI
        self.filesystem = FilesystemTools(
            SessionPermissions([self.workspace_root]),
            requester=self._request_filesystem_permission,
        )
        self.workspace_tools = WorkspaceReadTools(
            self.workspace_root,
            approver=self._edit_approver,
            command_approver=self._command_approver,
        )
        self.document_tools = DocumentTools(self.output_store.workspace_dir)
        try:
            vision_provider = registry.get_provider("qwen-vision")
        except KeyError:
            vision_provider = None
        self.vision_runtime = VisionRuntime(self.output_store.workspace_dir, vision_provider)
        self.tool_registry = self._build_tool_registry()
        self.policy_engine = build_policy_engine(
            self.tool_registry,
            self.workspace_root,
            approval_requester=self._policy_approval,
            audit=self.audit.log,
        )
        # These deterministic file tools predate ToolMeta permissions, so make
        # their side-effect policy explicit at the gateway boundary.
        for name in ("write_file", "create_directory"):
            policy = self.policy_engine.policy_for(name)
            if policy:
                from runtime.tool_policy import ToolPolicy
                self.policy_engine.register(ToolPolicy(
                    name=name, filesystem="workspace_only", requires_approval=True,
                    timeout=policy.timeout, max_output=policy.max_output,
                ))
        from runtime.tool_policy import ToolPolicy
        self.policy_engine.register(ToolPolicy(
            name="write_official_document", filesystem="deliverables_only",
            requires_approval=True, max_output=20_000,
        ))
        self.mcp_adapter = AegisMCPAdapter(self.tool_registry)
        # Additive capability-driven workflow seam; existing graph remains the
        # backwards-compatible default during migration.
        specialist_tools = {
                "read_file": self.workspace_tools.read_file,
                "list_directory": self.workspace_tools.list_directory,
                "tree": self.workspace_tools.tree,
                "search_files": self.workspace_tools.search_files,
                "find_files": self.workspace_tools.find_files,
                "get_file_info": self.workspace_tools.get_file_info,
                "repository_context": self.workspace_tools.repository_context,
                "git_status": self.workspace_tools.git_status,
                "git_diff": self.workspace_tools.git_diff,
                "edit_file": self.workspace_tools.edit_file,
                "create_file": self.workspace_tools.create_file,
                "create_python_script": self.workspace_tools.create_python_script,
                "execute_command": self.workspace_tools.execute_command,
                "document_runner": self._run_document_pipeline,
                "list_documents": self.document_tools.list_documents,
                "inspect_document_metadata": self.document_tools.inspect_document_metadata,
                "extract_document_text": self.document_tools.extract_document_text,
                "search_documents": self.document_tools.search_documents,
                "read_document_section": self.document_tools.read_document_section,
                "analyze_image": self.vision_runtime.analyze_image,
                "compare_images": self.vision_runtime.compare_images,
            }
        self.agent_registry = build_default_agent_registry(
            registry,
            tools=self.policy_engine.wrap_callables(specialist_tools),
        )
        try:
            master_provider = registry.get_provider("qwen-general")
        except KeyError:
            master_provider = None
        self.master_agent = MasterAgent(
            self.agent_registry,
            master_provider=master_provider,
            adaptive_router=self.adaptive_router,
            routing_mode=self.routing_mode,
        )
        self.master_agent.model_call_callback = self.network.record_model_call
        for specialist in self.agent_registry._agents.values():
            if hasattr(specialist, "model_call_callback"):
                specialist.model_call_callback = self.network.record_model_call
            if hasattr(specialist, "tool_call_callback"):
                specialist.tool_call_callback = self.network.record_tool_call
        self.graph = self._build_graph()

    def enable_adaptive_routing(self, adaptive_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
        """Enable adaptive selection only after the full non-inferiority gate."""
        self.adaptive_router.enable_after_evaluation(adaptive_report, baseline=baseline_report)
        self.routing_mode = "adaptive"
        self.master_agent.routing_mode = "adaptive"
        return {"enabled": True, "mode": "adaptive"}

    @staticmethod
    def _run_document_pipeline(input_path: str) -> dict[str, Any]:
        from pipeline.inspect_report import run_inspect_report
        return run_inspect_report(input_path=input_path)

    async def run_master(self, user_request: str, *, context: dict[str, Any] | None = None,
                         progress_callback: Any | None = None) -> dict[str, Any]:
        """Run every Master-first request through the universal task graph."""
        request_context = dict(context or {})
        request_context.setdefault("workspace_root", str(self.workspace_tools.root))
        # Preserve explicit local media/file references for the universal graph
        # instead of letting specialist selection discard them.
        path_match = re.search(r"(?:^|\s)([^\s]+\.(?:pdf|docx?|png|jpe?g|webp|bmp|tiff?|pgm|ppm|py|js|ts|rs|go|java|c|cpp))\b", user_request, re.I)
        if path_match:
            candidate = path_match.group(1).strip('`\"\'.,')
            if candidate.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pgm", ".ppm")):
                request_context.setdefault("image_paths", [candidate])
            elif candidate.lower().endswith((".pdf", ".doc", ".docx")):
                request_context.setdefault("input_path", candidate)
            else:
                request_context.setdefault("path", candidate)
        previous_callback = self.master_agent.progress_callback
        self.master_agent.progress_callback = progress_callback
        previous_command_callback = getattr(self.workspace_tools, "command_event_callback", None)
        run_id = self.output_store.new_run_id()
        started = time.perf_counter()
        self.workspace_tools.command_event_callback = progress_callback
        try:
            graph_state = await run_task_graph(
                self.master_agent,
                user_request,
                workspace_root=str(self.workspace_tools.root),
                progress_callback=progress_callback,
                max_task_retries=1,
                request_context=request_context,
                state_store=self.graph_state_store,
                run_id=run_id,
            )
        finally:
            self.master_agent.progress_callback = previous_callback
            self.workspace_tools.command_event_callback = previous_command_callback
        result_items = graph_state.get("step_results", [])
        status = "success" if graph_state.get("status") == "completed" else "failure"
        task_data = graph_state.get("task", {}) or {}
        selected_model_role = graph_state.get("selected_model", "")
        selected_model_id = ""
        if selected_model_role:
            try:
                selected_model_id = self.registry.get_provider(selected_model_role).model_id
            except (KeyError, AttributeError):
                selected_model_id = str(selected_model_role)
        eligible_candidates = task_data.get("eligible_candidates", []) or []
        alternative_models = [str(candidate.get("model")) for candidate in eligible_candidates
                              if isinstance(candidate, dict) and candidate.get("model")
                              and str(candidate.get("model")) != str(selected_model_role)]
        network_report = self.network.report()
        telemetry = self.telemetry_store.record(
            run_id=run_id, task_type=str(task_data.get("domain_intent") or task_data.get("operation") or "general"),
            required_capabilities=list(task_data.get("required_capabilities", []) or []),
            selected_model=str(selected_model_role),
            alternative_models=alternative_models,
            latency_ms=(time.perf_counter() - started) * 1000,
            token_count=int(graph_state.get("token_count", 0) or 0),
            tool_calls=int(graph_state.get("tool_call_count", 0) or 0),
            tool_failures=len(graph_state.get("errors", []) or []),
            verification_result=bool(graph_state.get("verification", {}).get("passed")),
            human_approval=bool(task_data.get("requires_human_approval")),
            final_success=graph_state.get("status") == "completed",
            metadata={"workflow": task_data.get("workflow", "general_reasoning"),
                      "routing": task_data.get("routing", {})},
            routing_context=task_data.get("routing_context", {}),
            eligible_candidates=task_data.get("eligible_candidates", []),
            selected_workflow=str(task_data.get("workflow", "general_reasoning")),
            routing_mode=str(task_data.get("routing_mode", self.routing_mode)),
            adaptive_fallback=bool(task_data.get("adaptive_fallback", False)),
            escalation=bool(graph_state.get("escalation_level", 0)),
            reward=reward_for_outcome(
                success=graph_state.get("status") == "completed",
                verification=bool(graph_state.get("verification", {}).get("passed")),
                evidence_quality=1.0 if graph_state.get("verification", {}).get("passed") else 0.0,
                human_accepted=not bool(task_data.get("requires_human_approval")) or bool(graph_state.get("approvals")),
                latency_ms=(time.perf_counter() - started) * 1000,
                failures=min(1.0, len(graph_state.get("errors", []) or []) / 5.0),
                escalated=bool(graph_state.get("escalation_level", 0)),
                unnecessary_escalation=bool(graph_state.get("escalation_level", 0)) and not bool(task_data.get("requires_human_approval")),
            ),
        )
        decision = task_data.get("bandit_decision")
        if isinstance(decision, dict):
            self.adaptive_router.record_observed_outcome(
                decision, task_data.get("routing_context", {}), telemetry["reward"],
                latency_ms=telemetry["latency_ms"],
                success=telemetry["final_success"],
                verification=telemetry["verification_result"],
                invalid=bool(telemetry["security_violations"] or telemetry["unauthorized_tool_executions"]
                             or telemetry["approval_bypasses"]),
            )
        metadata = {
            "run_id": run_id, "execution_mode": "master", "status": status,
            "task_type": "master", "modality": "text", "model": selected_model_id,
            "model_role": selected_model_role,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "telemetry": telemetry,
        }
        trace = [{"timestamp": datetime.now(timezone.utc).isoformat(), "step": e.get("event", "master"),
                  "message": e.get("event", "master"), "metadata": e} for e in graph_state.get("trace", [])]
        trace.append(make_trace_dict("network_status", "Network snapshot recorded", metadata=network_report))
        trace.append(make_trace_dict("master_persist", "Master result persisted", metadata={"run_id": run_id}))
        final_answer = graph_state.get("final_answer", "")
        output_dir = self.output_store.save_run(run_id=run_id, result=final_answer, metadata=metadata, trace=trace)
        (output_dir / "network_report.json").write_text(
            json.dumps(network_report, indent=2) + "\n", encoding="utf-8"
        )
        result = {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "execution_mode": "master",
            "status": graph_state.get("status", "failed"),
            "final_answer": final_answer,
            "preprocessing": graph_state.get("preprocessing", {}),
            "agent_results": result_items,
            "master_plan": graph_state.get("plan", []),
            "verification": graph_state.get("verification", {}),
            "errors": graph_state.get("errors", []),
            "repair_history": graph_state.get("repair_history", []),
            "selected_agent": graph_state.get("selected_agent", ""),
            "selected_model": graph_state.get("selected_model", ""),
            "task": graph_state.get("task", {}),
            "routing": graph_state.get("routing", {}),
            "checkpoint_store": str(self.workspace_root / ".aegis" / "runs.db"),
            "checkpoint_ref": graph_state.get("checkpoint_ref", ""),
            "telemetry": telemetry,
            "network_report": network_report,
            "trace": trace,
        }
        return result
    # -- Tool setup -----------------------------------------------------

    def _build_tool_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(calculator, tags=["math", "calculation"])
        reg.register(read_file, tags=["files", "io"])
        reg.register(write_file, tags=["files", "io"])
        reg.register(list_files, tags=["files", "io"])
        reg.register(create_directory, tags=["files", "io"])
        # Explicit capability only: do not change the existing image → vision route.
        reg.register(extract_ocr, tags=["ocr", "document_processing"])
        for filesystem_tool in self.filesystem.as_langchain_tools():
            if filesystem_tool.name in {"list_directory", "find_files", "inspect_path", "read_file", "safe_read_file", "write_file", "edit_file"}:
                continue
            reg.register(filesystem_tool, requires_approval=True, tags=["filesystem", "local"])
        for workspace_tool in self.workspace_tools.as_langchain_tools():
            mutation = workspace_tool.name in {
                "edit_file", "create_file", "create_python_script", "execute_command",
                "create_checkpoint", "restore_checkpoint",
            }
            reg.register(workspace_tool, requires_approval=mutation,
                        tags=["workspace", "mutation" if mutation else "read_only"],
                        permissions={"read": not mutation, "write": mutation, "delete": workspace_tool.name == "restore_checkpoint",
                                     "network": False, "external_side_effect": False,
                                     "credential_access": False, "system_access": False},
                        reversible=workspace_tool.name not in {"execute_command"})
        for document_tool in self.document_tools.as_langchain_tools():
            reg.register(document_tool, tags=["document", "local", "read_only"])
        for vision_tool in self.vision_runtime.as_langchain_tools():
            reg.register(vision_tool, tags=["vision", "local", "read_only"])
        reg.register(self._official_document_tool(), requires_approval=True,
                     tags=["document", "official", "approval"],
                     permissions={"read": False, "write": True, "delete": False,
                                  "network": False, "external_side_effect": False,
                                  "credential_access": False, "system_access": False},
                     reversible=False)
        return reg

    def _official_document_tool(self):
        workflow = self.official_documents

        @tool("write_official_document")
        def write_official_document(subject: str, background: str, proposal: str,
                                    financial_implication: str, dop_authority: str,
                                    recommendation: str, filename: str = "official_approval_note.docx") -> dict[str, Any]:
            """Generate an official approval DOCX after the policy gate approves it."""
            note = ApprovalNote(subject=subject, background=background, proposal=proposal,
                                financial_implication=financial_implication,
                                dop_authority=dop_authority, recommendation=recommendation)
            state = workflow.prepare(note)
            # The surrounding PolicyEngine is the human gate. Reaching this
            # callable means that gate has already approved the action.
            return workflow.generate(note, request_id=state.request_id, approve=True, filename=filename)

        return write_official_document

    def _request_filesystem_permission(self, mode: AccessMode, path: Path, reason: str) -> bool:
        """Ask in the terminal; grants are held only by this Orchestrator instance."""
        action = f"{mode.value.upper()} access requested"
        details = f"Path:\n {path}\n\nReason:\n{reason}"
        approved = self.approvals.prompt_terminal("filesystem-" + datetime.now(timezone.utc).strftime("%f"), action, details)
        self.audit.log("filesystem_permission", action=mode.value, status="approved" if approved else "rejected",
                       metadata={"path": str(path), "reason": reason})
        return approved

    def _edit_approver(self, action: str, path: str, old_text: str = "", new_text: str = "") -> bool:
        """Bounded interactive approval for workspace file mutations."""
        from runtime.tool_policy import policy_approval_granted
        if policy_approval_granted():
            return True
        if self._approval_fn is not None:
            return bool(self._approval_fn(action, path, old_text, new_text))
        action_name = f"{action} {path}"
        details = (f"old_text: {len(old_text)} chars → new_text: {len(new_text)} chars"
                   if action == "edit_file" else f"content: {len(old_text)} chars")
        task_id = "edit-" + datetime.now(timezone.utc).strftime("%f")
        approved = self.approvals.prompt_terminal(task_id, action_name, details)
        self.audit.log(
            "edit_approval",
            status="approved" if approved else "denied",
            metadata={"path": path, "old_len": len(old_text), "new_len": len(new_text)},
        )
        return approved

    def _command_approver(self, action: str, command: str, cwd: str = ".") -> bool:
        """Bounded interactive approval for execute_command. Shows command and cwd."""
        from runtime.tool_policy import policy_approval_granted
        if policy_approval_granted():
            return True
        if self._approval_fn is not None:
            return bool(self._approval_fn(action, command, cwd))
        action_name = f"execute_command in '{cwd}'"
        details = f"command: {command[:200]}"
        task_id = "cmd-" + datetime.now(timezone.utc).strftime("%f")
        approved = self.approvals.prompt_terminal(task_id, action_name, details)
        self.audit.log(
            "command_approval",
            status="approved" if approved else "denied",
            metadata={"command": command[:200], "cwd": cwd},
        )
        return approved

    def _policy_approval(self, request_id: str, tool: str, details: str) -> bool:
        """Human checkpoint used by the policy gateway, separate from retries."""
        if self._approval_fn is not None:
            return bool(self._approval_fn(tool, details))
        action = tool
        if tool in {"edit_file", "create_file", "create_python_script"}:
            match = re.search(r"(?:path|file_path)=([^,]+)", details)
            if match:
                action = f"{tool} {match.group(1)}"
        return self.approvals.prompt_terminal(request_id, action, details)

    def _legacy_tool_approval(self, request_id: str, tool: str, details: str) -> bool:
        """Adapt legacy workspace approvers behind the central policy gate."""
        approver = getattr(self.workspace_tools, "approver", None)
        command_approver = getattr(self.workspace_tools, "command_approver", None)
        if tool in {"edit_file", "create_file", "create_python_script"} and approver:
            path = re.search(r"(?:path|file_path)=([^,]+)", details)
            content = re.search(r"content=([^,]+)", details)
            return bool(approver(tool, path.group(1) if path else "", "", content.group(1) if content else ""))
        if tool == "execute_command" and command_approver:
            command = re.search(r"command=([^,]+)", details)
            cwd = re.search(r"cwd=([^,]+)", details)
            return bool(command_approver(tool, command.group(1) if command else "", cwd.group(1) if cwd else "."))
        return self._policy_approval(request_id, tool, details)

    # -- Graph construction ---------------------------------------------

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(AgentState)

        builder.add_node("classify", self._classify_node)
        builder.add_node("route", self._route_node)
        builder.add_node("execute_tool", self._execute_tool_node)
        builder.add_node("execute_model", self._execute_model_node)

        builder.add_edge(START, "classify")
        builder.add_edge("classify", "route")

        # Conditional edge: direct tool execution vs model execution
        builder.add_conditional_edges(
            "route",
            self._route_decision,
            {
                "execute_tool": "execute_tool",
                "execute_model": "execute_model",
            },
        )

        builder.add_edge("execute_tool", END)
        builder.add_edge("execute_model", END)

        return builder.compile(checkpointer=MemorySaver())

    # -- Conditional router ---------------------------------------------

    @staticmethod
    def _route_decision(state: AgentState) -> str:
        """Branch between direct tool execution and model execution."""
        if state.get("is_direct_tool"):
            return "execute_tool"
        return "execute_model"

    # -- Nodes ----------------------------------------------------------

    async def _classify_node(self, state: AgentState) -> dict[str, Any]:
        from routing.classifier import Task
        t0 = time.perf_counter()
        last_msg = state["messages"][-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        attached = state.get("attached_files", [])

        task: Task = self.classifier.classify(
            user_text,
            attached,
            previous_task_type=state.get("previous_task_type"),  # NEW
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        self.audit.log(
            "task_classified", task_id=task.id,
            metadata={
                "task_type": task.task_type.value, "modality": task.modality.value,
                "complexity": task.complexity.value, "execution_mode": task.execution_mode.value,
            },
        )

        task_dict = task.to_serializable_dict()

        return {
            "task": task_dict,
            "attached_files": task.attached_files,
            "execution_mode": task.execution_mode.value,
            "classification_ms": elapsed_ms,
            "current_step": "classified",
            "previous_task_type": task.task_type.value,  # NEW: carried forward
            "trace": [
                make_trace_dict(
                    step="classify",
                    message=f"type={task.task_type.value}  modality={task.modality.value}  mode={task.execution_mode.value}",
                    duration_ms=elapsed_ms,
                )
            ],
        }

    async def _route_node(self, state: AgentState) -> dict[str, Any]:
        """Select the best model or direct tool for the task."""
        from routing.classifier import Task
        from routing.router import RoutingResult
        t0 = time.perf_counter()

        task_dict = state["task"]
        task: Task = Task.model_validate(task_dict)
        result: RoutingResult = self.router.route(task, availability=self.availability)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.audit.log(
            "model_routed",
            task_id=task.id,
            model=result.model_id,
            metadata={
                "score": result.score,
                "reason": result.reason,
                "is_direct_tool": result.is_direct_tool,
            },
        )

        trace_msg = (
            f"Direct tool: {task.direct_tool_name}"
            if result.is_direct_tool
            else f"→ {result.model_name} ({result.model_id})  score={result.score:.1f}"
        )

        return {
            "selected_model": result.model_name,
            "selected_model_id": result.model_id,
            "routing_reason": result.reason,
            "is_direct_tool": result.is_direct_tool,
            "routing_ms": elapsed_ms,
            "current_step": "routed",
            "trace": [
                make_trace_dict(
                    step="route",
                    message=trace_msg,
                    duration_ms=elapsed_ms,
                )
            ],
        }

    async def _execute_tool_node(self, state: AgentState) -> dict[str, Any]:
        """Execute a deterministic tool directly without invoking an LLM."""
        t0 = time.perf_counter()

        task_dict = state["task"]
        tool_name = task_dict.get("direct_tool_name")
        tool_args = task_dict.get("direct_tool_args", {})

        if not tool_name:
            error_msg = "Direct tool execution requested, but tool name was not specified."
            return {
                "messages": [AIMessage(content=f"Error: {error_msg}")],
                "errors": [error_msg],
                "current_step": "error",
            }

        self.network.record_tool_call(tool=tool_name, local=True)

        try:
            # All direct tool calls cross the same policy gateway as model
            # tool calls.  The implementation remains in the registry.
            override = read_file if tool_name == "read_file" else write_file if tool_name == "write_file" else None
            tool_result = await self.policy_engine.execute(
                self.tool_registry, tool_name, tool_args,
                task_id=str(task_dict.get("id", "")),
                tool_override=override,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            self.audit.log(
                "tool_executed",
                task_id=task_dict.get("id"),
                tool=tool_name,
                status="success",
                duration_ms=elapsed_ms,
                metadata={"args": tool_args},
            )

            total_ms = round(
                state.get("classification_ms", 0.0)
                + state.get("routing_ms", 0.0)
                + elapsed_ms,
                2,
            )

            return {
                "messages": [AIMessage(content=str(tool_result))],
                "tool_ms": elapsed_ms,
                "total_ms": total_ms,
                "current_step": "completed",
                "trace": [
                    make_trace_dict(
                        step="tool",
                        message=f"Executed {tool_name} successfully ({elapsed_ms:.1f}ms)",
                        duration_ms=elapsed_ms,
                    )
                ],
            }

        except PolicyDenied as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            error_msg = f"Tool '{tool_name}' blocked by policy: {exc}"
            self.audit.log("tool_executed", task_id=task_dict.get("id"), tool=tool_name,
                           status="policy_denied", error=str(exc), duration_ms=elapsed_ms)
            return {"messages": [AIMessage(content=f"Error: {error_msg}")],
                    "errors": [error_msg], "tool_ms": elapsed_ms,
                    "current_step": "blocked", "trace": [make_trace_dict(
                        step="policy", message=error_msg, duration_ms=elapsed_ms)]}
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            error_msg = f"Tool '{tool_name}' execution error: {exc}"
            self.audit.log(
                "tool_executed",
                task_id=task_dict.get("id"),
                tool=tool_name,
                status="failure",
                error=str(exc),
                duration_ms=elapsed_ms,
            )
            return {
                "messages": [AIMessage(content=f"Error: {error_msg}")],
                "errors": [error_msg],
                "tool_ms": elapsed_ms,
                "current_step": "error",
                "trace": [
                    make_trace_dict(
                        step="tool",
                        message=error_msg,
                        duration_ms=elapsed_ms,
                    )
                ],
            }

    async def _stream_tool_loop(
        self,
        provider: Any,
        messages: list[Any],
        encoded_images: list[str] | None = None,
        allowed_tools: set[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Compatibility event adapter over the native bounded coding graph."""
        user_task = next(
            (str(message.content) for message in reversed(messages)
             if isinstance(message, HumanMessage)),
            "",
        )
        graph_messages = [SystemMessage(content=_TOOL_LOOP_PROMPT.format(
            tools=", ".join(sorted(allowed_tools or self.tool_registry.list_names())),
            task=user_task[:4000]
        )), *messages]
        try:
            final_state = await run_coding_graph(
                provider, self.tool_registry,
                str(messages[-1].content if messages else ""),
                allowed_tools=allowed_tools,
                messages=graph_messages,
                encoded_images=encoded_images or [],
                max_iterations=self.max_iterations,
                on_model_call=lambda: self.network.record_model_call(model=provider.model_id, local=True),
                on_tool_call=lambda: self.network.record_tool_call(tool="coding_graph", local=True),
                policy_engine=self.policy_engine,
                approval_callback=self._legacy_tool_approval,
            )
        except Exception as exc:
            yield {"kind": "error", "message": f"Coding graph error: {exc}"}
            return
        for event in final_state.get("events", []):
            kind = event.get("kind")
            if kind == "tool_requested":
                yield {"kind": "tool", "tool": event.get("tool"),
                       "arguments": event.get("arguments", {}),
                       "iteration": event.get("iteration", 0)}
            elif kind == "tool_result":
                yield {"kind": "result", "tool": event.get("tool"),
                       "result": event.get("result", ""),
                       "iteration": event.get("iteration", 0),
                       "arguments": event.get("arguments", {}),
                       "status": event.get("status", "failure"),
                       "duration_ms": event.get("duration_ms", 0),
                       "result_size": event.get("result_size", 0),
                       "truncated": event.get("truncated", False),
                       **({"approval": event["approval"]} if event.get("approval") else {}),
                       **({"verification": True} if event.get("verification") else {})}
            elif kind == "safe_error":
                yield {"kind": "error", "message": final_state.get("final_answer", "Coding graph stopped safely.")}
        if final_state.get("terminal_status") == "success":
            yield {"kind": "final", "answer": final_state.get("final_answer", ""),
                   "iteration": final_state.get("iteration", 0), "duration_ms": 0}
        elif final_state.get("terminal_status") != "success" and not final_state.get("final_answer"):
            yield {"kind": "error", "message": "; ".join(final_state.get("errors", [])) or "Coding graph stopped safely."}
        return

    async def run_tool_sequence(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute an explicit, ordered local tool plan without bypassing policy.

        Each result remains primitive-only, and one failed tool does not abort a
        later independent call.  This is the bridge used by the terminal agent
        for multi-step local analysis; model routing itself remains unchanged.
        """
        results: list[dict[str, Any]] = []
        for call in calls:
            name = str(call.get("tool", ""))
            args = call.get("args", {})
            if not isinstance(args, dict):
                results.append({"ok": False, "tool": name, "error": "InvalidArguments"})
                continue
            try:
                self.network.record_tool_call(tool=name, local=True)
                output = await self.policy_engine.execute(
                    self.tool_registry, name, args, task_id="tool_sequence"
                )
                results.append({"ok": True, "tool": name, "result": output})
            except PolicyDenied as exc:
                # Preserve the legacy explicit-sequence contract for an
                # unknown registry name while retaining the policy decision
                # and denial audit event. Registered tools still expose the
                # canonical PolicyDenied failure to callers.
                if name not in self.tool_registry.list_names():
                    results.append({"ok": False, "tool": name, "error": "KeyError"})
                else:
                    results.append({"ok": False, "tool": name, "error": type(exc).__name__, "message": str(exc)})
            except Exception as exc:
                results.append({"ok": False, "tool": name, "error": type(exc).__name__, "message": str(exc)})
        return results

    async def _execute_model_node(self, state: AgentState) -> dict[str, Any]:
        """Invoke the selected model (reasoning, coding, or vision)."""
        t0 = time.perf_counter()

        model_name: str = state["selected_model"]
        provider = self.registry.get_provider(model_name)
        task_dict = state.get("task", {})

        # Prepare messages
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=_SYSTEM_PROMPT))

        # Handle image attachments for vision tasks
        attached_files = state.get("attached_files", []) or task_dict.get("attached_files", [])
        image_files = [
            f for f in attached_files
            if Path(f).suffix.lower() in (
                ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".svg"
            )
        ]

        self.network.record_model_call(model=provider.model_id, local=True)

        try:
            image_t0 = time.perf_counter()
            encoded_images = provider.encode_images(image_files) if image_files else []
            image_load_ms = round((time.perf_counter() - image_t0) * 1000, 2)
            response_data = await provider.chat(
                self._provider_messages(messages), encoded_images=encoded_images
            )
            response = AIMessage(content=response_data.content)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            self.audit.log(
                "model_invoked",
                task_id=task_dict.get("id"),
                model=provider.model_id,
                status="success",
                duration_ms=elapsed_ms,
            )

            total_ms = round(
                state.get("classification_ms", 0.0)
                + state.get("routing_ms", 0.0)
                + elapsed_ms,
                2,
            )

            return {
                "messages": [response],
                "model_ms": elapsed_ms,
                "image_load_ms": image_load_ms,
                "total_ms": total_ms,
                "current_step": "completed",
                "trace": [
                    make_trace_dict(
                        step="execute",
                        message=f"Response generated ({len(str(response.content))} chars, {elapsed_ms:.0f}ms)",
                        duration_ms=elapsed_ms,
                    )
                ],
            }

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            error_msg = f"Model error: {exc}"
            self.audit.log(
                "model_invoked",
                model=provider.model_id,
                status="failure",
                error=str(exc),
                duration_ms=elapsed_ms,
            )
            return {
                "messages": [AIMessage(content=f"Error: {error_msg}")],
                "current_step": "error",
                "errors": [error_msg],
                "model_ms": elapsed_ms,
                "trace": [
                    make_trace_dict(
                        step="execute",
                        message=error_msg,
                        duration_ms=elapsed_ms,
                    )
                ],
            }

    @staticmethod
    def _provider_messages(messages: list[Any]) -> list[dict[str, Any]]:
        """Convert LangChain messages to Ollama's primitive chat payload schema."""
        converted: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                role = "user"
            converted.append({"role": role, "content": str(message.content)})
        return converted

    def _persist_run(self, state: AgentState) -> dict[str, Any]:
        """Persist a complete run without recording prompt/document content."""
        started_at = state.get("run_started_at") or datetime.now(timezone.utc).isoformat()
        task = state.get("task") or {}
        result = ""
        if state.get("messages"):
            result = str(state["messages"][-1].content)
        run_id = self.output_store.new_run_id()
        status = "failure" if state.get("errors") or state.get("current_step") == "error" else "success"
        metadata = {
            "run_id": run_id, "task_type": task.get("task_type"), "modality": task.get("modality"),
            "execution_mode": state.get("execution_mode"), "model": state.get("selected_model_id"),
            "input_files": [Path(f).name for f in state.get("attached_files", [])], "status": status,
            "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": state.get("total_ms"),
        }
        network_report = self.network.report()
        trace = list(state.get("trace", []))
        trace.append(make_trace_dict("network_status", "Network snapshot recorded", metadata=network_report))
        trace.append(make_trace_dict("persist_output", "Output persisted", metadata={"run_id": run_id}))
        output_dir = self.output_store.save_run(run_id=run_id, result=result, metadata=metadata, trace=trace)
        (output_dir / "network_report.json").write_text(
            json.dumps(network_report, indent=2) + "\n", encoding="utf-8"
        )
        return {"run_id": run_id, "output_dir": str(output_dir), "trace": trace}

    # -- Public interface -----------------------------------------------

    async def astream(
        self,
        user_input: str,
        thread_id: str,
        *,
        attached_files: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one request with native provider token streaming for the terminal."""
        from routing.classifier import ExecutionMode
        initial_state: dict[str, Any] = {
            "messages": [HumanMessage(content=user_input)],
            "attached_files": attached_files or [],
            "iteration": 0,
            "run_started_at": datetime.now(timezone.utc).isoformat(),
            "previous_task_type": self._session_task_type.get(thread_id),  # NEW
        }
        classify_out = await self._classify_node(initial_state)
        state = {**initial_state, **classify_out}
        self._session_task_type[thread_id] = classify_out["task"]["task_type"]  # NEW
        yield {"event": "on_chain_end", "name": "classify", "data": {"output": classify_out}}
        route_out = await self._route_node(state)
        prior_trace = list(state.get("trace", []))
        route_trace = route_out.get("trace", [])
        state.update(route_out)
        state["trace"] = prior_trace + route_trace
        yield {"event": "on_chain_end", "name": "route", "data": {"output": route_out}}

        if state.get("is_direct_tool"):
            output = await self._execute_tool_node(state)
            output["trace"] = list(state.get("trace", [])) + output.get("trace", [])
            state.update(output)
            persisted = self._persist_run(state)
            output.update(persisted)
            yield {"event": "on_chain_end", "name": "execute_tool", "data": {"output": output}}
            return

        task = state.get("task", {})
        image_files = [
            f for f in state.get("attached_files", [])
            if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".svg")
        ]
        provider = self.registry.get_provider(state["selected_model"])
        messages = list(state["messages"])
        messages.insert(0, SystemMessage(content=_SYSTEM_PROMPT))
        started = time.perf_counter()
        try:
            image_t0 = time.perf_counter()
            encoded_images = provider.encode_images(image_files) if image_files else []
            image_load_ms = round((time.perf_counter() - image_t0) * 1000, 2)

            if state.get("execution_mode") == ExecutionMode.MODEL_WITH_TOOLS.value:
                final_answer = ""
                loop_error = ""
                tool_trace: list[dict[str, Any]] = []
                loop_started = time.perf_counter()
                workspace_allowed = {"list_directory", "tree", "read_file", "search_files", "find_files", "get_file_info", "repository_context", "git_status", "git_diff", "edit_file", "create_file", "create_python_script", "execute_command"}
                allowed_tools = workspace_allowed if "workspace_read" in task.get("requires_tools", []) else None
                async for loop_event in self._stream_tool_loop(provider, messages, encoded_images, allowed_tools):
                    kind = loop_event.get("kind")
                    if kind == "tool":
                        tool_trace.append(make_trace_dict("tool", f"Invoked {loop_event.get('tool', 'unknown')}"))
                        yield {"event": "on_agent_status", "name": "tool", "data": loop_event}
                    elif kind == "result":
                        meta = {
                            "step": loop_event.get("iteration"),
                            "model": state.get("selected_model_id") or getattr(provider, "model_id", "qwen2.5-coder:7b"),
                            "tool": loop_event["tool"],
                            "arguments": loop_event.get("arguments", {}),
                            "status": loop_event["status"],
                            "duration_ms": loop_event["duration_ms"],
                            "result_size": loop_event["result_size"],
                            "truncated": loop_event["truncated"],
                        }
                        if "approval" in loop_event:
                            meta["approval"] = loop_event["approval"]
                        tool_trace.append(make_trace_dict(
                            "tool_result", f"{loop_event['tool']} {loop_event['status']}",
                            loop_event["duration_ms"],
                            metadata=meta,
                        ))
                        yield {"event": "on_agent_status", "name": "tool_result", "data": loop_event}
                    elif kind == "status":
                        yield {"event": "on_agent_status", "name": "agent", "data": loop_event}
                    elif kind == "final":
                        final_answer = str(loop_event.get("answer", ""))
                        yield {"event": "on_chat_model_stream", "name": "AgentFinal", "data": {"chunk": AIMessage(content=final_answer)}}
                    elif kind == "error":
                        loop_error = str(loop_event.get("message", "Agent loop failed."))
                        yield {"event": "on_agent_status", "name": "agent", "data": loop_event}
                elapsed_ms = round((time.perf_counter() - loop_started) * 1000, 2)
                failed = bool(loop_error)
                if loop_error and not final_answer:
                    final_answer = f"Error: {loop_error}"
                output = {
                    "messages": [AIMessage(content=final_answer)], "model_ms": elapsed_ms,
                    "image_load_ms": image_load_ms, "model_first_token_ms": None,
                    "total_ms": round(state.get("classification_ms", 0) + state.get("routing_ms", 0) + elapsed_ms, 2),
                    "current_step": "error" if failed else "completed",
                    "errors": [loop_error] if failed else [],
                    "trace": tool_trace + [make_trace_dict("agent_loop", "Tool loop completed", elapsed_ms)],
                }
                output["trace"] = list(state.get("trace", [])) + output["trace"]
                state.update(output)
                persisted = self._persist_run(state)
                output.update(persisted)
                yield {"event": "on_chain_end", "name": "execute_model", "data": {"output": output}}
                return

            first_token_ms: float | None = None
            response_parts: list[str] = []
            self.network.record_model_call(model=provider.model_id, local=True)
            stream_method = getattr(provider, "stream_chat_events", None)
            stream = stream_method(self._provider_messages(messages), encoded_images=encoded_images,
                                   think=os.getenv("SHOW_OLLAMA_THINKING", "0") == "1") if stream_method else provider.stream_chat(
                                       self._provider_messages(messages), encoded_images=encoded_images)
            async for token_event in stream:
                token = token_event.get("token", "") if isinstance(token_event, dict) else token_event
                if isinstance(token_event, dict) and token_event.get("kind") == "thinking":
                    yield {"event": "on_agent_status", "name": "thinking", "data": {"token": token}}
                    continue
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000, 2)
                response_parts.append(token)
                yield {"event": "on_chat_model_stream", "name": "OllamaProvider", "data": {"chunk": AIMessage(content=token)}}
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            total_ms = round(state.get("classification_ms", 0) + state.get("routing_ms", 0) + elapsed_ms, 2)
            output = {
                "messages": [AIMessage(content="".join(response_parts))], "model_ms": elapsed_ms,
                "image_load_ms": image_load_ms, "model_first_token_ms": first_token_ms,
                "total_ms": total_ms, "current_step": "completed",
                "trace": [make_trace_dict("execute", f"Response generated ({len(''.join(response_parts))} chars, {elapsed_ms:.0f}ms)", elapsed_ms),
                          make_trace_dict("model_first_token", "First visible token received", first_token_ms)],
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            output = {
                "messages": [AIMessage(content=f"Error: Model error: {exc}")], "errors": [f"Model error: {exc}"],
                "model_ms": elapsed_ms, "current_step": "error",
                "trace": [make_trace_dict("execute", f"Model error: {exc}", elapsed_ms)],
            }
        output["trace"] = list(state.get("trace", [])) + output.get("trace", [])
        state.update(output)
        persisted = self._persist_run(state)
        output.update(persisted)
        yield {"event": "on_chain_end", "name": "execute_model", "data": {"output": output}}

    async def ainvoke(
        self,
        user_input: str,
        thread_id: str,
        *,
        attached_files: list[str] | None = None,
    ) -> AgentState:
        """Run the graph to completion and return the final state."""
        from routing.classifier import ExecutionMode
        task = self.classifier.classify(
            user_input,
            attached_files or [],
            previous_task_type=self._session_task_type.get(thread_id),  # NEW
        )
        if task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS:
            collected: dict[str, Any] = {}
            async for event in self.astream(user_input, thread_id, attached_files=attached_files):
                if event.get("event") == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        collected.update(output)
            return collected  # type: ignore[return-value]

        initial_state: dict[str, Any] = {
            "messages": [HumanMessage(content=user_input)],
            "attached_files": attached_files or [],
            "iteration": 0,
            "run_started_at": datetime.now(timezone.utc).isoformat(),
            "previous_task_type": self._session_task_type.get(thread_id),  # NEW
        }
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 20,
        }

        result = await self.graph.ainvoke(initial_state, config=config)
        self._session_task_type[thread_id] = result.get("task", {}).get("task_type")  # NEW
        persisted = self._persist_run(result)
        result.update(persisted)
        return result
