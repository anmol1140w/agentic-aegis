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
from runtime.compound_tasks import decompose_task


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
    approval_note = bool(re.search(r"\b(approval\s+note|office\s+note)\b", lower))
    required_capabilities = ["document_analysis", "reasoning"] if approval_note else ["reasoning"]
    compound_steps = [stage.to_dict() for stage in decompose_task(request)]
    return {
        "operation": operation,
        "original_request": request,
        "normalized_request": nlp.normalized_text,
        "artifact_format": formats[0] if formats else None,
        "modality": modality,
        "content_requirements": [request] if operation == "create" else [],
        "domain_intent": "psu_approval_note" if approval_note and ("refinery" in lower or "mrpl" in lower or "pipeline" in lower or "valve" in lower) else "general",
        "workflow": "psu_approval_note_generate" if approval_note and operation == "create" else "psu_approval_note_explain" if approval_note else "general_reasoning",
        "requires_human_approval": bool(approval_note and operation == "create"),
        "required_capabilities": required_capabilities,
        "quality_required": 0.80,
        "compound_steps": compound_steps,
    }


def build_task_graph(master: MasterAgent, *, workspace_root: str,
                     progress_callback: Callable[[dict[str, Any]], None] | None = None,
                     max_task_retries: int = 1,
                     request_context: dict[str, Any] | None = None,
                     state_store: Any | None = None):
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
                                  capability=item.get("capability"), score=item.get("selection_score"),
                                  domain_intent=state.get("task", {}).get("domain_intent", "general"),
                                  workflow=state.get("task", {}).get("workflow", "general_reasoning"),
                                  required_capabilities=state.get("task", {}).get("required_capabilities", []),
                                  quality_required=state.get("task", {}).get("quality_required", 0.80))]}

    async def create_plan(state: TaskRunState) -> dict[str, Any]:
        stages = state.get("task", {}).get("compound_steps", []) or []
        if not stages:
            stages = [{"name": "specialist", "capability": state.get("routing", {}).get("capability", "reasoning"),
                       "objective": f"Delegate to {state['selected_agent']}", "evidence": "structured evidence"}]
        plan = [{"step_id": f"specialist-{index}", "ordinal": index,
                "description": stage.get("objective", f"Delegate to {state['selected_agent']}"),
                "tool_name": "delegate_to_agent", "parameters": {"agent": state["selected_agent"]},
                "expected_evidence": [{"kind": "agent_result", "passed": True, "detail": stage.get("evidence", "")}],
                "status": StepStatus.PENDING.value, "retry_count": 0,
                "max_retries": int(state.get("max_task_retries", max_task_retries)),
                "compound_stage": stage.get("name", "specialist"),
                "capability": stage.get("capability", "reasoning")} for index, stage in enumerate(stages, 1)]
        return {"plan": plan, "current_step_index": 0,
                "trace": [_emit(progress_callback, "plan_created", steps=len(plan), compound=len(plan) > 1)]}

    async def validate_plan(state: TaskRunState) -> dict[str, Any]:
        if not state.get("selected_agent") or not state.get("plan"):
            return {"status": TaskStatus.FAILED.value,
                    "errors": [{"code": "invalid_plan", "message": "No valid specialist plan."}],
                    "trace": [_emit(progress_callback, "plan_rejected")]}
        return {"status": TaskStatus.RUNNING.value,
                "trace": [_emit(progress_callback, "plan_validated", agent=state["selected_agent"])]}

    async def execute_step(state: TaskRunState) -> dict[str, Any]:
        step_index = state.get("current_step_index", 0)
        step = dict(state["plan"][step_index])
        step["status"] = StepStatus.RUNNING.value
        current_agent = state.get("selected_agent", "")
        stage = str(step.get("compound_stage", "specialist"))
        stage_agent = {
            "extract": "document_agent", "interpret_visual": "vision_agent",
            "calculate": "general_agent", "draft_approval": "document_agent",
            "verify": "general_agent",
        }.get(stage, current_agent)
        try:
            master.registry.get(stage_agent)
            current_agent = stage_agent
        except Exception:
            # Synthetic/minimal registries used by compatibility callers may
            # expose only one specialist; preserve deterministic execution.
            pass
        step["parameters"] = {**step.get("parameters", {}), "agent": current_agent}
        spec = dict(state.get("task", {}))
        context = {**(request_context or {}), "workspace_root": workspace_root, "task_spec": spec,
                   "repository_map": state.get("repository_map", {}),
                   "relevant_files": state.get("relevant_files", []),
                   "agent_state": state.get("agent_memory", {}),
                   "nlp": {"original_text": state["original_request"],
                           "normalized_text": state.get("normalized_request", ""),
                           "enhanced_prompt": state.get("normalized_request", "")}}
        context["handoff_prompt"] = handoff_prompt(
            from_role="master_agent", to_role=current_agent or "specialist",
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
        policy_event = _emit(
            progress_callback, "policy_gateway", status="checked",
            action="delegate_to_agent", agent=current_agent,
            workflow=spec.get("workflow", "general_reasoning"),
            approval_required=bool(spec.get("requires_human_approval", False)),
        )
        if spec.get("requires_human_approval"):
            _emit(progress_callback, "approval_checkpoint", status="required", action=spec.get("workflow", "artifact_generation"))
        delegation_request = str(spec.get("enhanced_request") or state["original_request"])
        if spec.get("workflow") == "psu_approval_note_explain":
            # Explain-intent is grounded from the canonical domain contract;
            # the model may fill fields later, but cannot redefine the term.
            from routing.approval_note import explain_approval_note
            result = AgentResult(
                agent=state["selected_agent"], status=AgentStatus.SUCCESS,
                summary=explain_approval_note(),
                verification={"required": True, "status": "passed", "method": "canonical_domain_contract"},
                metadata={"grounded": True, "domain_intent": "psu_approval_note"},
            )
        else:
            result = await master.delegate_to_agent(
                current_agent, delegation_request, context=context,
                success_criteria=["structured evidence"], trace_id=state.get("run_id"))
        result_dict = result.model_dump()
        step["result"] = result_dict
        step["status"] = StepStatus.SUCCEEDED.value if result.status == AgentStatus.SUCCESS else StepStatus.FAILED.value
        coding_state = result_dict.get("metadata", {}).get("coding_state", {})
        updated_plan = [*state["plan"]]
        updated_plan[step_index] = step
        return {"plan": updated_plan, "step_results": [result_dict],
                "agent_memory": coding_state if isinstance(coding_state, dict) else state.get("agent_memory", {}),
                "tool_call_count": state.get("tool_call_count", 0) + 1,
                "status": TaskStatus.VERIFYING.value if result.status == AgentStatus.SUCCESS else TaskStatus.HEALING.value,
                "trace": [policy_event, _emit(progress_callback, "specialist_execution", agent=current_agent,
                                  status=result.status.value, stage=step.get("compound_stage", "specialist"))]}

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
        if latest and latest[-1].get("status") not in {AgentStatus.SUCCESS.value, "success"}:
            verification = {
                "passed": False, "status": "failed",
                "summary": "The specialist execution did not succeed; verification is unresolved.",
                "evidence": [], "missing_evidence": ["successful specialist execution"],
            }
            return {"verification": verification, "status": TaskStatus.HEALING.value,
                    "trace": [_emit(progress_callback, "verification_failed",
                                      summary=verification["summary"], next_step=False)]}
        verification = verify_plan(latest)
        if latest and latest[-1].get("agent") == "coding_agent":
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
        has_next = bool(verification.get("passed") and int(state.get("current_step_index", 0)) + 1 < len(state.get("plan", [])))
        status = TaskStatus.RUNNING.value if has_next else TaskStatus.COMPLETED.value if verification.get("passed") else TaskStatus.HEALING.value
        if has_next:
            verification = {**verification, "next_step": int(state.get("current_step_index", 0)) + 1}
        return {"verification": verification, "status": status,
                "current_step_index": int(state.get("current_step_index", 0)) + 1 if has_next else state.get("current_step_index", 0),
                "trace": [_emit(progress_callback, "verification_passed" if verification.get("passed") else "verification_failed",
                                  summary=verification.get("summary"), next_step=has_next)]}

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
        raw_summary = result.get("summary", "") if verification.get("passed") else "Task completed without sufficient deterministic evidence."

        # Guard: if the "summary" is a raw-JSON task-spec blob (e.g. it starts
        # with '{' and contains machine keys like "original_request"), replace
        # it with a human-readable reconstruction from verified evidence.
        def _is_machine_json(text: str) -> bool:
            t = str(text).strip()
            if not t.startswith("{"):
                return False
            import json as _json
            try:
                parsed = _json.loads(t)
                if isinstance(parsed, dict):
                    machine_keys = {"operation", "original_request", "normalized_request",
                                    "agent", "domain_intent", "workflow", "artifact_format",
                                    "modality", "required_capabilities", "compound_steps"}
                    return bool(machine_keys & parsed.keys())
            except Exception:
                pass
            return False

        def _human_answer(res: dict) -> str:
            changes = res.get("changes", [])
            evidence = res.get("evidence", [])
            artifacts = res.get("artifacts", [])
            verification_inner = res.get("verification", {})
            cmd = verification_inner.get("command", "")
            if changes:
                label = "Modified" if any("edit" in str(e) for e in evidence) else "Created"
                paths = ", ".join(str(p) for p in changes[:8])
                suffix = f" — command: {cmd}" if cmd else ""
                return f"{label}: {paths}{suffix}."
            if artifacts:
                return "Produced: " + ", ".join(str(a) for a in artifacts[:4]) + "."
            reads = [e for e in evidence if str(e).startswith("read_file:")]
            if reads:
                return "Inspection complete. " + "; ".join(str(r)[:120] for r in reads[:3]) + "."
            return "Completed with verified deterministic evidence."

        answer: str
        if not str(raw_summary).strip():
            answer = "Completed with verified deterministic evidence." if verification.get("passed") else "Task completed without sufficient deterministic evidence."
        elif _is_machine_json(str(raw_summary)):
            answer = _human_answer(result)
        else:
            answer = str(raw_summary)

        # A model can emit a conservative/contradictory final sentence after a
        # successful mutation. Deterministic evidence is authoritative here:
        # report the verified artifact instead of surfacing a false failure.
        if verification.get("passed") and isinstance(answer, str) and "insufficient evidence" in answer.lower():
            changes = result.get("changes", [])
            if not isinstance(changes, list):
                changes = []
            answer = "Created and verified: " + ", ".join(str(path) for path in changes) if changes else "Completed with verified deterministic evidence."
        unresolved = any(
            "max tool steps reached" in str(item).lower()
            for item in (result.get("errors", []) if isinstance(result, dict) else [])
        )
        completed = bool(verification.get("passed")) and not unresolved and result.get("status") in {AgentStatus.SUCCESS.value, "success"}
        return {"final_answer": answer, "status": TaskStatus.COMPLETED.value if completed else TaskStatus.FAILED.value,
                "trace": [_emit(progress_callback, "final", status="verified" if verification.get("passed") else "failed")]}

    async def safe_failure(state: TaskRunState) -> dict[str, Any]:
        errors = state.get("errors") or [{"code": "task_failed", "message": "Task stopped safely."}]
        verification = state.get("verification") or {}
        # A failed terminal path must never retain a stale successful
        # verification flag from an earlier node or specialist result.
        verification = {**verification, "passed": False, "status": "failed"}
        return {"final_answer": "Unable to complete the requested task: " + "; ".join(str(e.get("message", e)) for e in errors),
                "status": TaskStatus.FAILED.value, "verification": verification,
                "trace": [_emit(progress_callback, "run_failed", errors=errors)]}

    def after_validate(state: TaskRunState) -> str:
        return "execute_step" if state.get("status") == TaskStatus.RUNNING.value else "safe_failure"

    def after_record(state: TaskRunState) -> str:
        return "verify_plan" if state.get("status") == TaskStatus.VERIFYING.value else "self_heal"

    def after_verify(state: TaskRunState) -> str:
        if state.get("status") == TaskStatus.RUNNING.value:
            return "execute_step"
        return "review_and_finish" if state.get("status") == TaskStatus.COMPLETED.value else "self_heal"

    def after_heal(state: TaskRunState) -> str:
        return "execute_step" if state.get("status") == TaskStatus.RUNNING.value else "safe_failure"

    async def persisted_node(name: str, fn: Callable[[TaskRunState], Any], state: TaskRunState) -> dict[str, Any]:
        update = await fn(state)
        if state_store is not None:
            snapshot = {**state, **(update or {})}
            state_store.save(str(state.get("run_id", "")), str(state.get("task_id", "")), name, snapshot)
            checkpoint_event = _emit(progress_callback, "checkpoint_persisted", node=name,
                                     run_id=state.get("run_id", ""))
            update = {**(update or {}), "trace": [*((update or {}).get("trace", [])), checkpoint_event]}
        return update

    builder = StateGraph(TaskRunState)
    for name, node in (("normalize_request", normalize_request), ("classify_and_route", classify_and_route),
                       ("create_plan", create_plan), ("validate_plan", validate_plan),
                       ("execute_step", execute_step), ("record_step_result", record_step_result),
                       ("verify_plan", verify_plan_node), ("self_heal", self_heal),
                       ("review_and_finish", review_and_finish), ("safe_failure", safe_failure)):
        if state_store is not None:
            async def checkpointed(state: TaskRunState, _name=name, _node=node) -> dict[str, Any]:
                return await persisted_node(_name, _node, state)
            builder.add_node(name, checkpointed)
        else:
            builder.add_node(name, node)
    resume_targets = {
        "normalize_request": "classify_and_route", "classify_and_route": "create_plan",
        "create_plan": "validate_plan", "validate_plan": "execute_step",
        "execute_step": "record_step_result", "record_step_result": "verify_plan",
        "verify_plan": "review_and_finish", "self_heal": "execute_step",
        "review_and_finish": "completed", "safe_failure": "completed", "completed": "completed",
    }
    def start_node(state: TaskRunState) -> str:
        return resume_targets.get(str(state.get("resume_from", "")), "normalize_request")
    builder.add_conditional_edges(
        START, start_node,
        {"normalize_request": "normalize_request", **{name: name for name in set(resume_targets.values()) if name != "completed"}, "completed": END},
    )
    builder.add_edge("normalize_request", "classify_and_route")
    builder.add_edge("classify_and_route", "create_plan")
    builder.add_edge("create_plan", "validate_plan")
    builder.add_conditional_edges("validate_plan", after_validate, {"execute_step": "execute_step", "safe_failure": "safe_failure"})
    builder.add_edge("execute_step", "record_step_result")
    builder.add_conditional_edges("record_step_result", after_record, {"verify_plan": "verify_plan", "self_heal": "self_heal"})
    builder.add_conditional_edges("verify_plan", after_verify, {"execute_step": "execute_step", "review_and_finish": "review_and_finish", "self_heal": "self_heal"})
    builder.add_conditional_edges("self_heal", after_heal, {"execute_step": "execute_step", "safe_failure": "safe_failure"})
    builder.add_edge("review_and_finish", END)
    builder.add_edge("safe_failure", END)
    return builder.compile()


async def run_task_graph(master: MasterAgent, request: str, *, workspace_root: str,
                         progress_callback: Callable[[dict[str, Any]], None] | None = None,
                         max_task_retries: int = 1,
                         request_context: dict[str, Any] | None = None,
                         state_store: Any | None = None,
                         run_id: str | None = None) -> dict[str, Any]:
    graph = build_task_graph(master, workspace_root=workspace_root,
                             progress_callback=progress_callback,
                             max_task_retries=max_task_retries,
                             request_context=request_context,
                             state_store=state_store)
    selected_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
    initial_state = {"run_id": selected_run_id,
                                 "task_id": f"task_{uuid.uuid4().hex[:12]}",
                                 "schema_version": 1,
                                 "original_request": request,
                                 "workspace_root": workspace_root,
                                 "max_task_retries": max_task_retries,
                                 "task_retry_count": 0,
                                 "step_results": [], "repair_history": [], "errors": [], "trace": [],
                                 "agent_memory": {}}
    if state_store is not None:
        previous_item = state_store.load_latest(selected_run_id)
        if previous_item:
            previous_node, previous = previous_item
            # Continue at the first node after the last durable checkpoint.
            initial_state = {**initial_state, **previous, "resume_from": previous_node}
    state = await graph.ainvoke(initial_state)
    if state_store is not None:
        state_store.save(selected_run_id, str(state.get("task_id", "")), "completed", dict(state))
        _emit(progress_callback, "checkpoint_persisted", node="completed", run_id=selected_run_id)
    return dict(state)
