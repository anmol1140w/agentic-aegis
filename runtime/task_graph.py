"""Universal bounded task graph for MasterAgent specialist workflows.

This graph is the single execution boundary for document, vision, coding, and
lightweight specialist calls.  Specialists still own their tools and policy;
the graph owns sequencing, evidence, verification, and bounded repair.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from runtime.agents import AgentRequest, AgentResult, AgentStatus, MasterAgent
from runtime.nlp import NLPPreprocessor
from runtime.self_healing import classify_failure, normalize_repair_action
from runtime.task_state import TaskRunState, TaskStatus, StepStatus
from runtime.verification import verify_coding_result, verify_plan
from runtime.repository_index import RepositoryIndex
from runtime.context_manager import ContextBudgetManager
from runtime.model_profiles import get_model_profile
from runtime.reviewer import deterministic_review
from runtime.prompts import handoff_prompt


def _emit(callback: Callable[[dict[str, Any]], None] | None, event: str, **data: Any) -> dict[str, Any]:
    item = {"event": event, "timestamp": time.time(), **data}
    if callback:
        callback(item)
    return item


def _task_spec(request: str, nlp: Any) -> dict[str, Any]:
    """Conservative task metadata; content remains the original request."""
    text = nlp.enhanced_prompt or nlp.normalized_text or request
    lower = text.lower()
    operation = "create" if re.search(r"\b(create|write|generate|make|produce)\b", lower) else "analyze"
    formats = re.findall(r"\b(docx|pdf|markdown|md|txt)\b", lower)
    modality = "image" if re.search(r"\b(image|p&id|diagram|visual)\b", lower) else "document" if re.search(r"\b(pdf|document|report|ocr|markdown|md|txt)\b", lower) else "text"
    return {
        "operation": operation,
        "original_request": request,
        "normalized_request": nlp.normalized_text,
        "artifact_format": formats[0] if formats else None,
        "modality": modality,
        "content_requirements": [request] if operation == "create" else [],
    }


def build_task_graph(master: MasterAgent, *, workspace_root: str,
                     progress_callback: Callable[[dict[str, Any]], None] | None = None,
                     max_task_retries: int = 1,
                     request_context: dict[str, Any] | None = None):
    """Compile the universal Master → specialist graph."""

    async def normalize_request(state: TaskRunState) -> dict[str, Any]:
        nlp = NLPPreprocessor().process(state["original_request"])
        spec = _task_spec(state["original_request"], nlp)
        supplied = (request_context or {}).get("task_spec")
        if isinstance(supplied, dict):
            spec.update(supplied)
        repository_map: dict[str, Any] = {}
        try:
            index = RepositoryIndex(workspace_root)
            index.refresh()
            repository_map = index.summary()
        except Exception as exc:
            repository_map = {"root": workspace_root, "index_error": str(exc)}
        return {"normalized_request": nlp.enhanced_prompt,
                "preprocessing": nlp.to_dict(),
                "task": spec,
                "repository_root": workspace_root,
                "repository_map": repository_map,
                "facts": [f"repository root: {workspace_root}"],
                "status": TaskStatus.PENDING.value,
                "trace": [_emit(progress_callback, "preprocessing_completed",
                                  normalization_count=nlp.metadata.get("normalization_count", 0),
                                  entity_count=nlp.metadata.get("entity_count", 0))]}

    async def classify_and_route(state: TaskRunState) -> dict[str, Any]:
        request = state["normalized_request"] or state["original_request"]
        if hasattr(master, "route_request"):
            planned = await master.route_request(request, context=request_context or {})
        else:
            planned = master._capability_plan(request)
        if not planned:
            return {"status": TaskStatus.FAILED.value,
                    "errors": [{"code": "no_capability", "message": "No suitable capability found."}],
                    "trace": [_emit(progress_callback, "capability_discovery_failed")]}
        item = planned[0]
        relevant_files: list[str] = []
        try:
            index = RepositoryIndex(workspace_root)
            relevant_files = index.relevant_files(request)
        except Exception:
            pass
        agent_name = str(item.get("agent", ""))
        try:
            descriptor = master.registry.get(agent_name).descriptor
        except Exception:
            return {"status": TaskStatus.FAILED.value,
                    "errors": [{"code": "agent_unavailable", "message": agent_name}],
                    "trace": [_emit(progress_callback, "capability_discovery_failed", agent=agent_name)]}
        return {"selected_agent": agent_name, "selected_model": descriptor.provider_name,
                "relevant_files": relevant_files,
                "routing": {k: v for k, v in item.items() if k not in {"task", "agent"}},
                "task": {**state.get("task", {}),
                          "enhanced_request": str(item.get("task", request)),
                          **{k: v for k, v in item.items() if k not in {"task", "agent"}}},
                "status": TaskStatus.PLANNED.value,
                "trace": [_emit(progress_callback, "capability_discovery", agent=agent_name,
                                  capability=item.get("capability"), score=item.get("selection_score"))]}

    async def create_plan(state: TaskRunState) -> dict[str, Any]:
        step = {"step_id": "specialist-1", "ordinal": 1,
                "description": f"Delegate to {state['selected_agent']}",
                "tool_name": "delegate_to_agent", "parameters": {"agent": state["selected_agent"]},
                "expected_evidence": [{"kind": "agent_result", "passed": True}],
                "status": StepStatus.PENDING.value, "retry_count": 0,
                "max_retries": int(state.get("max_task_retries", max_task_retries))}
        return {"plan": [step], "current_step_index": 0,
                "trace": [_emit(progress_callback, "plan_created", steps=1)]}

    async def validate_plan(state: TaskRunState) -> dict[str, Any]:
        if not state.get("selected_agent") or not state.get("plan"):
            return {"status": TaskStatus.FAILED.value,
                    "errors": [{"code": "invalid_plan", "message": "No valid specialist plan."}],
                    "trace": [_emit(progress_callback, "plan_rejected")]}
        return {"status": TaskStatus.RUNNING.value,
                "trace": [_emit(progress_callback, "plan_validated", agent=state["selected_agent"])]}

    async def execute_step(state: TaskRunState) -> dict[str, Any]:
        step = dict(state["plan"][state.get("current_step_index", 0)])
        step["status"] = StepStatus.RUNNING.value
        spec = dict(state.get("task", {}))
        context = {**(request_context or {}), "workspace_root": workspace_root, "task_spec": spec,
                   "repository_map": state.get("repository_map", {}),
                   "relevant_files": state.get("relevant_files", []),
                   "agent_state": state.get("agent_memory", {}),
                   "nlp": {"original_text": state["original_request"],
                           "normalized_text": state.get("normalized_request", ""),
                           "enhanced_prompt": state.get("normalized_request", "")}}
        context["handoff_prompt"] = handoff_prompt(
            from_role="master_agent", to_role=state.get("selected_agent", "specialist"),
            task=state["original_request"], objective=step.get("description", "complete the current step"),
            state={"state_version": state.get("current_step_index", 0),
                   "repository_map": state.get("repository_map", {}),
                   "relevant_files": state.get("relevant_files", [])},
            observation=state.get("observations", [])[-1:],
            allowed_tools=state.get("relevant_files", []),
        )
        bundle = ContextBudgetManager().build(
            system="AEGIS execution contract: use approved tools, provide evidence, and stop on policy failures.",
            task=state["original_request"], facts=state.get("facts", []),
            observation=state.get("observations", [])[-1:], tools=state.get("relevant_files", []),
        )
        context["context_bundle"] = bundle.text
        context["context_metadata"] = {"prompt_chars": bundle.prompt_chars, "truncated": bundle.truncated,
                                        "included_files": bundle.included_files}
        delegation_request = str(spec.get("enhanced_request") or state["original_request"])
        result = await master.delegate_to_agent(
            state["selected_agent"], delegation_request, context=context,
            success_criteria=["structured evidence"], trace_id=state.get("run_id"))
        result_dict = result.model_dump()
        step["result"] = result_dict
        step["status"] = StepStatus.SUCCEEDED.value if result.status == AgentStatus.SUCCESS else StepStatus.FAILED.value
        coding_state = result_dict.get("metadata", {}).get("coding_state", {})
        return {"plan": [step], "step_results": [result_dict],
                "agent_memory": coding_state if isinstance(coding_state, dict) else state.get("agent_memory", {}),
                "tool_call_count": state.get("tool_call_count", 0) + 1,
                "status": TaskStatus.VERIFYING.value if result.status == AgentStatus.SUCCESS else TaskStatus.HEALING.value,
                "trace": [_emit(progress_callback, "specialist_execution", agent=state["selected_agent"],
                                  status=result.status.value)]}

    async def record_step_result(state: TaskRunState) -> dict[str, Any]:
        result = (state.get("step_results") or [{}])[-1]
        if result.get("status") in {AgentStatus.SUCCESS.value, "success"}:
            return {"trace": [_emit(progress_callback, "step_succeeded", agent=state.get("selected_agent"))]}
        failure = classify_failure(result)
        detail = result.get("errors")
        if isinstance(detail, list) and detail:
            detail = str(detail[0])
        else:
            detail = result.get("summary", "specialist failed")
        return {"errors": [{**failure, "message": detail}],
                "trace": [_emit(progress_callback, "step_failed", agent=state.get("selected_agent"),
                                  error_code=failure.get("error_code"))]}

    async def verify_plan_node(state: TaskRunState) -> dict[str, Any]:
        # A repaired step supersedes its failed attempt for final verification;
        # the complete history remains available in ``step_results`` and trace.
        latest = state.get("step_results", [])[-1:]
        verification = verify_plan(latest)
        if state.get("selected_agent") == "coding_agent" and latest:
            coding_check = verify_coding_result(latest[-1], state.get("original_request", ""))
            verification = {
                **coding_check,
                "evidence": [*verification.get("evidence", []), *coding_check.get("evidence", [])],
            }
        if latest:
            review = deterministic_review(workspace_root=workspace_root, result=latest[-1], verification=verification)
            verification = {**verification, "review": review.model_dump(),
                            "passed": bool(verification.get("passed") and review.passed)}
        # Artifact-producing specialists must carry their own deterministic
        # verification contract; reject a prose-only success.
        if verification.get("passed") and latest:
            item = latest[-1]
            nested = item.get("result") if isinstance(item, dict) else None
            nested_ver = nested.get("verification") if isinstance(nested, dict) else None
            if isinstance(nested_ver, dict) and nested_ver.get("required") and nested_ver.get("status") not in {"passed", "verified"}:
                verification = {**verification, "passed": False, "status": "failed",
                                "summary": "artifact verification did not pass",
                                "missing_evidence": ["verified artifact"]}
        status = TaskStatus.COMPLETED.value if verification.get("passed") else TaskStatus.HEALING.value
        return {"verification": verification, "status": status,
                "trace": [_emit(progress_callback, "verification_passed" if verification.get("passed") else "verification_failed",
                                  summary=verification.get("summary"))]}

    async def self_heal(state: TaskRunState) -> dict[str, Any]:
        attempts = int(state.get("task_retry_count", 0))
        result = (state.get("step_results") or [{}])[-1]
        failure = classify_failure(result)
        if not failure["retryable"]:
            return {
                "status": TaskStatus.FAILED.value,
                "errors": [{**failure, "message": "Repair is not permitted for this policy or path failure."}],
                "trace": [_emit(progress_callback, "repair_rejected",
                                  reason="non_retryable", error_code=failure["error_code"])],
            }
        if attempts >= int(state.get("max_task_retries", max_task_retries)):
            return {"status": TaskStatus.FAILED.value,
                    "errors": [{"code": "repair_budget_exhausted", "message": "No bounded repair remains."}],
                    "trace": [_emit(progress_callback, "repair_rejected", reason="budget_exhausted")]}
        # A repair is structured and capability-preserving: retry the same
        # specialist only for a retryable failure. No unrelated fallback agent.
        decision = normalize_repair_action({"action": "retry", "reason": "retryable specialist failure"})
        return {"task_retry_count": attempts + 1, "repair_history": [decision],
                "status": TaskStatus.RUNNING.value,
                "trace": [_emit(progress_callback, "repair_requested", action=decision["action"],
                                  agent=state.get("selected_agent"), error=result.get("errors"))]}

    async def review_and_finish(state: TaskRunState) -> dict[str, Any]:
        verification = state.get("verification", {})
        result = (state.get("step_results") or [{}])[-1]
        answer = result.get("summary", "") if verification.get("passed") else "Task completed without sufficient deterministic evidence."
        return {"final_answer": answer, "status": TaskStatus.COMPLETED.value if verification.get("passed") else TaskStatus.FAILED.value,
                "trace": [_emit(progress_callback, "final", status="verified" if verification.get("passed") else "failed")]}

    async def safe_failure(state: TaskRunState) -> dict[str, Any]:
        errors = state.get("errors") or [{"code": "task_failed", "message": "Task stopped safely."}]
        return {"final_answer": "Unable to complete the requested task: " + "; ".join(str(e.get("message", e)) for e in errors),
                "status": TaskStatus.FAILED.value,
                "trace": [_emit(progress_callback, "run_failed", errors=errors)]}

    def after_validate(state: TaskRunState) -> str:
        return "execute_step" if state.get("status") == TaskStatus.RUNNING.value else "safe_failure"

    def after_record(state: TaskRunState) -> str:
        return "verify_plan" if state.get("status") == TaskStatus.VERIFYING.value else "self_heal"

    def after_verify(state: TaskRunState) -> str:
        return "review_and_finish" if state.get("status") == TaskStatus.COMPLETED.value else "self_heal"

    def after_heal(state: TaskRunState) -> str:
        return "execute_step" if state.get("status") == TaskStatus.RUNNING.value else "safe_failure"

    builder = StateGraph(TaskRunState)
    for name, node in (("normalize_request", normalize_request), ("classify_and_route", classify_and_route),
                       ("create_plan", create_plan), ("validate_plan", validate_plan),
                       ("execute_step", execute_step), ("record_step_result", record_step_result),
                       ("verify_plan", verify_plan_node), ("self_heal", self_heal),
                       ("review_and_finish", review_and_finish), ("safe_failure", safe_failure)):
        builder.add_node(name, node)
    builder.add_edge(START, "normalize_request")
    builder.add_edge("normalize_request", "classify_and_route")
    builder.add_edge("classify_and_route", "create_plan")
    builder.add_edge("create_plan", "validate_plan")
    builder.add_conditional_edges("validate_plan", after_validate, {"execute_step": "execute_step", "safe_failure": "safe_failure"})
    builder.add_edge("execute_step", "record_step_result")
    builder.add_conditional_edges("record_step_result", after_record, {"verify_plan": "verify_plan", "self_heal": "self_heal"})
    builder.add_conditional_edges("verify_plan", after_verify, {"review_and_finish": "review_and_finish", "self_heal": "self_heal"})
    builder.add_conditional_edges("self_heal", after_heal, {"execute_step": "execute_step", "safe_failure": "safe_failure"})
    builder.add_edge("review_and_finish", END)
    builder.add_edge("safe_failure", END)
    return builder.compile()


async def run_task_graph(master: MasterAgent, request: str, *, workspace_root: str,
                         progress_callback: Callable[[dict[str, Any]], None] | None = None,
                         max_task_retries: int = 1,
                         request_context: dict[str, Any] | None = None) -> dict[str, Any]:
    graph = build_task_graph(master, workspace_root=workspace_root,
                             progress_callback=progress_callback,
                             max_task_retries=max_task_retries,
                             request_context=request_context)
    state = await graph.ainvoke({"run_id": f"run_{uuid.uuid4().hex[:12]}",
                                 "task_id": f"task_{uuid.uuid4().hex[:12]}",
                                 "schema_version": 1,
                                 "original_request": request,
                                 "workspace_root": workspace_root,
                                 "max_task_retries": max_task_retries,
                                 "task_retry_count": 0,
                                 "step_results": [], "repair_history": [], "errors": [], "trace": [],
                                 "agent_memory": {}})
    return dict(state)
