"""Capability-driven specialist agents and bounded master orchestration.

This module is an additive migration seam.  The existing classifier/router
graph remains available while callers can opt into a capability-driven master
workflow through :class:`MasterAgent`.
"""

from __future__ import annotations

import json
import time
import uuid
import asyncio
import re
import tempfile
import os
import zipfile
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from runtime.nlp import NLPPreprocessor
from runtime.capability_matching import CapabilityMatcher, CapabilityProfile

from models.base import ModelProvider
from models.registry import ModelRegistry
from runtime.approvals import ApprovalManager
from runtime.actions import ActionParseError, parse_action
from tools.vision import VisionPreprocessor


class AgentStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class AgentCapability(str, Enum):
    CODING = "coding"
    VISION = "vision"
    DOCUMENT = "document_analysis"
    LIGHTWEIGHT = "lightweight"
    GENERAL = "general"
    VERIFICATION = "verification"


def _coerce_list_of_strings(val: Any) -> list[str]:
    """Ensure a string is wrapped in a single-element list rather than split by characters."""
    if isinstance(val, str):
        val_clean = val.strip()
        return [val_clean] if val_clean else []
    if isinstance(val, (list, tuple, set)):
        return [str(item) for item in val if str(item).strip()]
    if val is None:
        return []
    return [str(val)]


class AgentRequest(BaseModel):
    task: str
    context: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    expected_output: str = "structured result"
    success_criteria: list[str] = Field(default_factory=list)
    trace_id: str = ""
    depth: int = 0
    agent_execution_id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")

    @field_validator("success_criteria", "constraints", mode="before")
    @classmethod
    def _validate_string_lists(cls, v: Any) -> list[str]:
        return _coerce_list_of_strings(v)

    @field_validator("agent_execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, v: Any) -> str:
        if not v or not isinstance(v, str) or v.strip() == "{{agent_execution_id}}" or "{{" in v:
            return f"exec_{uuid.uuid4().hex[:12]}"
        return v.strip()


class AgentResult(BaseModel):
    agent: str = ""
    agent_execution_id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")
    status: AgentStatus
    summary: str = ""
    result: Any = None
    evidence: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    verification: dict[str, Any] = Field(default_factory=lambda: {
        "required": False, "command": "", "status": "not_run"
    })
    errors: list[str] = Field(default_factory=list)
    next_recommendation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def recommendation(self) -> str:
        return self.next_recommendation

    @field_validator("agent_execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, v: Any) -> str:
        if not v or not isinstance(v, str) or v.strip() == "{{agent_execution_id}}" or "{{" in v:
            return f"exec_{uuid.uuid4().hex[:12]}"
        return v.strip()


class AgentDescriptor(BaseModel):
    name: str
    role: str
    capabilities: list[AgentCapability]
    provider_name: str
    allowed_tools: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    output_schema: str = "AgentResult"
    modality: str = "text"
    description: str = ""
    available: bool = True
    permissions: list[str] = Field(default_factory=list)


class MasterTaskState(BaseModel):
    user_request: str
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    master_plan: list[dict[str, Any]] = Field(default_factory=list)
    subtasks: list[AgentRequest] = Field(default_factory=list)
    active_subtask: int | None = None
    agent_results: list[AgentResult] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    verification: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    final_answer: str = ""


class BaseAgent(ABC):
    """Specialist interface; implementations never choose another agent."""

    descriptor: AgentDescriptor

    @abstractmethod
    async def run(self, request: AgentRequest) -> AgentResult:
        raise NotImplementedError


class OllamaSpecialistAgent(BaseAgent):
    def __init__(self, descriptor: AgentDescriptor, provider: ModelProvider,
                 tools: dict[str, Callable[..., Any]] | None = None) -> None:
        self.descriptor = descriptor
        self.provider = provider
        self.tools = tools or {}

    async def run(self, request: AgentRequest) -> AgentResult:
        execution_id = request.agent_execution_id or f"exec_{uuid.uuid4().hex[:12]}"
        evidence: list[str] = []
        artifacts: list[str] = []
        changes: list[str] = []
        approvals: list[dict[str, Any]] = []
        verification: dict[str, Any] = {"required": False, "command": "", "status": "not_run"}
        # Creation is an explicit operation of the document specialist and
        # does not require an input document.
        if AgentCapability.DOCUMENT in self.descriptor.capabilities and re.search(
                r"\b(create|write|generate)\b.*\b(document|docx|pdf|markdown|md|txt|report)\b", request.task, re.I):
            spec = request.context.get("task_spec") if isinstance(request.context.get("task_spec"), dict) else {}
            topic_match = re.search(r"\b(?:about|of|on|for)\s+(.+?)(?=\s+(?:explain|describe|and\s+save|in\s+(?:docx|pdf|markdown|md|txt))|\s*$)", request.task, re.I)
            topic = str(spec.get("topic") or (topic_match.group(1).strip(" .\"'") if topic_match else "Requested topic"))
            requirements_value = spec.get("content_requirements")
            requirements = ("; ".join(map(str, requirements_value)) if isinstance(requirements_value, list) else str(requirements_value or ""))
            if not requirements:
                requirements = request.task[topic_match.end():].strip(" .\"") if topic_match else request.task
            format_match = re.search(r"\b(docx|pdf|markdown|md|txt)\b", request.task, re.I)
            requested_format = str(spec.get("artifact_format") or (format_match.group(1).lower() if format_match else "docx"))
            normalized_format = "markdown" if requested_format == "md" else requested_format
            output_root = Path(str(request.context.get("workspace_root") or Path.cwd())) / "outputs"
            try:
                output_root.mkdir(parents=True, exist_ok=True)
                safe_topic = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:60] or "document"
                out_dir = output_root / f"document_{uuid.uuid4().hex[:10]}"
                out_dir.mkdir(parents=True, exist_ok=False)
                artifact = out_dir / f"{safe_topic}.{('md' if normalized_format == 'markdown' else normalized_format)}"
                try:
                    generation_source = "local_model"
                    if self.provider is not None:
                        generation_prompt = (
                            "Write a concise, factual document in plain text with markdown-like headings. "
                            f"Topic: {topic}\n"
                            f"User request and content requirements: {request.task}\n"
                            f"Specific requirements: {requirements or 'provide a useful overview'}\n"
                            "Include a title, Introduction, 2-4 informative sections, and Conclusion. "
                            "Do not include meta-commentary or fabricate citations."
                        )
                        response = await asyncio.wait_for(self.provider.generate(generation_prompt), 90.0)
                        content = str(response.content or "").strip()
                    else:
                        generation_source = "bounded_template_no_provider"
                        content = (f"{topic}\n\n# Introduction\n\n"
                                   f"This document provides a concise overview of {topic}. {requirements}\n\n"
                                   f"# Background\n\nThe subject of {topic} is presented here in a structured, accessible format.\n\n"
                                   f"# Key points\n\nThis section summarizes important context and notable aspects of {topic}.\n\n"
                                   f"# Conclusion\n\nIn summary, {topic} remains a significant subject for further study.")
                    if not content.strip():
                        raise ValueError("content_generation_failed")
                    if normalized_format == "docx":
                        from pipeline.docx_writer import write_text_document
                        write_text_document(content, artifact)
                    else:
                        from pipeline.document_artifacts import create_document_artifact
                        create_document_artifact(content, normalized_format, artifact)
                except Exception as exc:
                    return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                       status=AgentStatus.FAILURE, summary="Document creation failed",
                                       errors=[str(exc)[:300]], metadata={"operation": "create_document"})
                content_length = len(content.strip())
                if normalized_format == "docx":
                    valid = artifact.exists() and artifact.stat().st_size > 0 and zipfile.is_zipfile(artifact)
                    readable = False
                    if valid:
                        try:
                            from docx import Document
                            readable = any(topic.lower() in p.text.lower() for p in Document(str(artifact)).paragraphs)
                        except Exception:
                            readable = False
                else:
                    from pipeline.document_artifacts import verify_artifact
                    checked = verify_artifact(artifact, normalized_format, expected_topic=topic)
                    valid, readable = checked.get("format_valid", False), checked.get("readable", False)
                verification = {"required": True, "exists": artifact.exists(), "readable": readable,
                                "format_valid": valid, "content_present": content_length > 0,
                                "content_length": content_length,
                                "status": "passed" if valid and readable and content_length > 0 else "failed"}
                if not (valid and readable):
                    return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                       status=AgentStatus.FAILURE, summary="Created document failed verification",
                                       verification=verification, errors=["artifact_verification_failed"],
                                       metadata={"operation": "create_document"})
                artifact_info = {"path": str(artifact), "format": normalized_format, "size_bytes": artifact.stat().st_size,
                                 "text_length": content_length}
                return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                   status=AgentStatus.SUCCESS, summary=f"Document created successfully for {topic}: {artifact}",
                                   result={"operation": "create_document", "artifact": artifact_info,
                                           "verification": verification}, artifacts=[str(artifact)],
                                   verification=verification, metadata={"operation": "create_document", "topic": topic,
                                                                         "requested_format": requested_format,
                                                                         "normalized_format": normalized_format,
                                                                         "generation_source": generation_source,
                                                                         "content_length": content_length,
                                                                         "content_requirements": requirements})
            except Exception as exc:
                return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                   status=AgentStatus.FAILURE, summary="Document creation failed",
                                   errors=[str(exc)[:300]], metadata={"operation": "create_document"})
        # Document processing stays in the established deterministic pipeline;
        # the specialist only adapts its structured output for Master review.
        if AgentCapability.DOCUMENT in self.descriptor.capabilities and "document_runner" in self.tools:
            source_path = request.context.get("input_path") or request.context.get("path")
            if not source_path:
                return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                   status=AgentStatus.FAILURE, summary="Document input path is required",
                                   errors=["missing input_path"])
            try:
                output = self.tools["document_runner"](source_path)
                if not isinstance(output, dict):
                    output = {"status": "complete", "result": output}
                if isinstance(output.get("ocr_text"), str):
                    ocr_nlp = NLPPreprocessor().process(output["ocr_text"], source="ocr")
                    # Keep raw OCR as evidence; expose normalized text only as
                    # an additional, bounded reasoning field.
                    output.setdefault("raw_ocr_text", output["ocr_text"])
                    output["normalized_ocr_text"] = ocr_nlp.normalized_text
                    output["ocr_nlp"] = ocr_nlp.to_dict()
                ok = output.get("status") in {"complete", "success"} or output.get("ok", False)
                return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                   status=AgentStatus.SUCCESS if ok else AgentStatus.FAILURE,
                                   summary="Document processed" if ok else "Document processing failed",
                                   result=output, evidence=[f"document:{Path(str(source_path)).name}"],
                                   artifacts=[str(p) for p in output.get("artifacts", [])] if isinstance(output.get("artifacts"), list) else [],
                                   errors=[] if ok else [str(output.get("error", "document failure"))],
                                   metadata={"input_path": Path(str(source_path)).name})
            except Exception as exc:
                return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                   status=AgentStatus.FAILURE, summary="Document processing failed",
                                   errors=[str(exc)[:500]])
        # Vision remains isolated to the configured provider and image payload;
        # timeout/cancellation errors are converted to structured failures.
        if AgentCapability.VISION in self.descriptor.capabilities:
            image_paths = request.context.get("image_paths") or request.context.get("images") or []
            vision_meta: dict[str, Any] = {"vision_calls": 1, "preprocessing_ms": 0.0}
            try:
                if not image_paths:
                    return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                       status=AgentStatus.FAILURE, summary="Vision image is required",
                                       errors=["missing image_paths"], metadata=vision_meta)
                with tempfile.TemporaryDirectory(prefix="vision_") as temp_dir:
                    prepared = VisionPreprocessor().process(str(image_paths[0]), Path(temp_dir) / "optimized.png")
                    vision_meta.update(prepared.metadata())
                    vision_meta["preprocessing_ms"] = prepared.processing_duration_ms
                    encoded = self.provider.encode_images([str(prepared.path)])
                    # Vision models can require several minutes on low-end
                    # GPUs. Keep this isolated/configurable so other agent
                    # deadlines remain unchanged while still bounding hangs.
                    vision_timeout = max(30.0, float(os.getenv("VISION_TIMEOUT_SECONDS", "360")))
                    response = await asyncio.wait_for(
                        self.provider.chat([{"role": "user", "content": request.task}], encoded_images=encoded),
                        vision_timeout,
                    )
                return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                   status=AgentStatus.SUCCESS if response.content.strip() else AgentStatus.FAILURE,
                                   summary=response.content[:4000], result=response.content,
                                   evidence=[f"image:{Path(str(p)).name}" for p in image_paths],
                                   metadata={"image_count": len(image_paths), **vision_meta})
            except asyncio.TimeoutError:
                return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                   status=AgentStatus.FAILURE, summary="Vision inference timed out",
                                   errors=["vision_timeout"], metadata=vision_meta)
            except Exception as exc:
                return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                   status=AgentStatus.FAILURE, summary="Vision inference failed",
                                   errors=[str(exc)[:500]], metadata=vision_meta)
        # Coding inspections must ground the model in source evidence.  Only
        # the explicitly allowlisted read_file tool is used in this phase.
        if AgentCapability.CODING in self.descriptor.capabilities and "read_file" in self.tools:
            candidate = request.context.get("path")
            if not candidate and re.search(r"\b(inspect|read|open|review|fix|edit|modify|change)\b", request.task, re.I):
                match = re.search(r"([\w./-]+\.(?:py|js|ts|rs|go|java|c|cpp|h))", request.task)
                candidate = match.group(1) if match else None
            if candidate:
                try:
                    reader = self.tools["read_file"]
                    source = reader(candidate)
                    if hasattr(source, "content"):
                        source = source.content
                    if hasattr(source, "invoke"):
                        source = source.invoke({"path": candidate})
                    if isinstance(source, dict) and source.get("ok") is False:
                        return AgentResult(agent_execution_id=execution_id, status=AgentStatus.FAILURE,
                                           summary="Unable to read requested source file", errors=[str(source.get("error", "read failed"))])
                    content = source.get("content", "") if isinstance(source, dict) else str(source)
                    evidence.append(f"read_file:{candidate} ({len(content)} chars)")
                    artifacts.append(str(candidate))
                except Exception as exc:
                    return AgentResult(agent_execution_id=execution_id, status=AgentStatus.FAILURE,
                                       summary="Unable to read requested source file", errors=[str(exc)[:500]])
        read_only_request = bool(re.search(r"\bdo not (?:modify|edit|change)\b", request.task, re.I))
        mutation_request = (not read_only_request) and bool(re.search(r"\b(fix|edit|modify|change|write|create|save|implement|pytest|run tests?)\b", request.task, re.I))
        if mutation_request and AgentCapability.CODING in self.descriptor.capabilities:
            # The model may propose actions, but infrastructure validates and
            # executes only descriptor-allowlisted tools. Approval is delegated
            # to WorkspaceReadTools; this layer never grants it implicitly.
            messages = [{"role": "user", "content": request.task}]
            saw_read = bool(evidence)
            invalid_actions = 0
            created_paths: set[str] = set()
            post_create_steps = 0
            for _ in range(8):
                if created_paths:
                    post_create_steps += 1
                    if post_create_steps > 4 and verification.get("status") == "verified" and not verification.get("command"):
                        return AgentResult(
                            agent=self.descriptor.name, agent_execution_id=execution_id,
                            status=AgentStatus.PARTIAL,
                            summary="File was created and verified, but the requested test command was not completed.",
                            evidence=evidence, artifacts=artifacts, changes=changes,
                            approvals=approvals, verification=verification,
                            errors=["test_execution_not_completed"],
                            next_recommendation="Run the approved test command after reviewing the verified file.",
                        )
                try:
                    response = await self.provider.generate(
                        json.dumps({
                            "task": request.task,
                            "evidence": evidence[-4:],
                            "available_tools": {
                                "read_file": {"required": ["path"]},
                                "list_directory": {"required": [], "optional": ["path"]},
                                "search_files": {"required": ["query"], "optional": ["path"]},
                                "find_files": {"required": ["pattern"], "optional": ["path"]},
                                "create_file": {"required": ["path", "content"]},
                                "create_python_script": {"required": ["path", "content"]},
                                "edit_file": {"required": ["path", "old_text", "new_text"]},
                                "execute_command": {"required": ["command"], "optional": ["cwd"]},
                            },
                            "instruction": (
                                "Return exactly one JSON object and no prose. For a workspace operation use "
                                '{"action":"tool","tool":"<allowed tool>","arguments":{...}}. '
                                "For a final response use {\"action\":\"final\",\"answer\":\"...\"}. "
                                "To create the requested program, call create_file before claiming success."
                            ),
                        })
                    )
                    action = parse_action(response.content)
                except ActionParseError as exc:
                    invalid_actions += 1
                    if invalid_actions < 3:
                        # Give the local model a bounded correction opportunity;
                        # free-form prose is never executed as a tool action.
                        evidence.append(f"invalid_action:{str(exc)[:160]}")
                        continue
                    return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                       status=AgentStatus.FAILURE,
                                       summary="Coding action was not valid", evidence=evidence,
                                       artifacts=artifacts, changes=changes, approvals=approvals,
                                       verification=verification,
                                       errors=[str(exc)[:500], "invalid_action_limit"])
                except Exception as exc:
                    return AgentResult(agent_execution_id=execution_id, status=AgentStatus.FAILURE,
                                       summary="Coding action was not valid", evidence=evidence,
                                       artifacts=artifacts, changes=changes, approvals=approvals,
                                       verification=verification, errors=[str(exc)[:500]])
                if action["action"] == "final":
                    if mutation_request and not changes:
                        return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                           status=AgentStatus.FAILURE,
                                           summary="No verified workspace change was produced",
                                           evidence=evidence, artifacts=artifacts,
                                           errors=["file_generation_unverified"])
                    if mutation_request and verification.get("required") and verification.get("status") not in {"verified", "passed"}:
                        return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id,
                                           status=AgentStatus.FAILURE,
                                           summary="Workspace change was not read back and verified",
                                           evidence=evidence, artifacts=artifacts, changes=changes,
                                           verification=verification, errors=["post_write_verification_required"])
                    return AgentResult(agent_execution_id=execution_id, status=AgentStatus.SUCCESS,
                                       summary=action["answer"], evidence=evidence, artifacts=artifacts,
                                       changes=changes, approvals=approvals, verification=verification)
                name, args = action["tool"], action.get("arguments", {})
                if name not in self.descriptor.allowed_tools or name not in self.tools:
                    return AgentResult(agent_execution_id=execution_id, status=AgentStatus.BLOCKED,
                                       summary="Tool is not permitted for this agent", evidence=evidence,
                                       errors=[f"unauthorized tool: {name}"])
                if name == "edit_file" and not saw_read:
                    return AgentResult(agent_execution_id=execution_id, status=AgentStatus.BLOCKED,
                                       summary="read_file is required before edit_file", evidence=evidence,
                                       errors=["edit-before-read blocked"])
                try:
                    result = self.tools[name](**args)
                except TypeError:
                    result = self.tools[name](args)
                if not isinstance(result, dict):
                    result = {"ok": True, "result": str(result)}
                status = result.get("status", "success" if result.get("ok", True) else "failure")
                if name == "read_file" and result.get("ok"):
                    saw_read = True
                    path = args.get("path", candidate or "")
                    content = str(result.get("content", ""))
                    evidence.append(f"read_file:{path} ({len(content)} chars)")
                    artifacts.append(str(path))
                    if verification.get("required"):
                        verification["status"] = "verified"
                elif name in {"edit_file", "create_file", "create_python_script"}:
                    approvals.append({"tool": name, "status": "approved" if status == "success" else "denied"})
                    if status == "success":
                        created_path = str(args.get("path", ""))
                        changes.append(created_path)
                        if name in {"create_file", "create_python_script"}:
                            created_paths.add(created_path)
                        verification = {"required": True, "command": "", "status": "verified_pending"}
                        # Infrastructure performs mandatory post-create
                        # readback; success never depends on the model
                        # remembering to verify its own write.
                        if "read_file" in self.tools:
                            try:
                                readback = self.tools["read_file"](created_path)
                                if isinstance(readback, dict) and readback.get("ok") and str(readback.get("content", "")).strip():
                                    saw_read = True
                                    verification["status"] = "verified"
                                    evidence.append(f"post_create_readback:{created_path}")
                                else:
                                    verification["status"] = "failed"
                            except Exception as exc:
                                verification["status"] = "failed"
                                evidence.append(f"post_create_readback_error:{str(exc)[:120]}")
                    elif (name in {"create_file", "create_python_script"}
                          and status != "success"
                          and result.get("error") == "file_exists"):
                        # Do not overwrite an existing file. Treat it as an
                        # artifact candidate and require readback/verification;
                        # this lets retries converge without masking content
                        # or path-safety failures.
                        existing_path = str(args.get("path", ""))
                        if existing_path not in changes:
                            changes.append(existing_path)
                        evidence.append(f"{name}:already_exists")
                        verification = {"required": True, "command": "", "status": "verified_pending"}
                        if "read_file" in self.tools:
                            try:
                                readback = self.tools["read_file"](existing_path)
                                if isinstance(readback, dict) and readback.get("ok") and str(readback.get("content", "")).strip():
                                    saw_read = True
                                    verification["status"] = "verified"
                                    evidence.append(f"post_create_readback:{existing_path}")
                                else:
                                    verification["status"] = "failed"
                            except Exception as exc:
                                verification["status"] = "failed"
                                evidence.append(f"post_create_readback_error:{str(exc)[:120]}")
                        continue
                elif name == "execute_command":
                    if verification.get("required") and verification.get("status") != "verified":
                        return AgentResult(agent_execution_id=execution_id, status=AgentStatus.BLOCKED,
                                           summary="Edited file must be verified before tests", evidence=evidence,
                                           changes=changes, approvals=approvals, verification=verification,
                                           errors=["post-edit verification required"])
                    approvals.append({"tool": name, "status": "approved" if result.get("approval") == "approved" or status == "success" else "denied"})
                    verification = {"required": True, "command": str(args.get("command", "")),
                                    "status": "passed" if status == "success" and result.get("exit_code", 0) == 0 else "failed"}
                evidence.append(f"{name}:{status}")
                if status != "success":
                    return AgentResult(agent_execution_id=execution_id, status=AgentStatus.FAILURE,
                                       summary=f"{name} failed", evidence=evidence, artifacts=artifacts,
                                       changes=changes, approvals=approvals, verification=verification,
                                       errors=[str(result.get("error", result.get("message", "tool failed")))])
            return AgentResult(agent_execution_id=execution_id, status=AgentStatus.BLOCKED,
                               summary="Coding tool loop limit reached", evidence=evidence,
                               artifacts=artifacts, changes=changes, approvals=approvals,
                               verification=verification, errors=["max tool steps reached"])
        prompt = (
            f"Role: {self.descriptor.role}\nTask: {request.task}\n"
            f"Context: {json.dumps(request.context, default=str)[:8000]}\n"
            f"Constraints: {request.constraints}\n"
            f"Evidence references: {evidence}\nExpected output: {request.expected_output}\n"
            "Return a JSON object matching AgentResult."
        )
        try:
            response = await self.provider.generate(prompt)
            raw = response.content.strip()
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    # Infrastructure/code controls agent_execution_id; model cannot invent or control it.
                    data["agent_execution_id"] = execution_id
                    result = AgentResult.model_validate(data)
                    result.agent = self.descriptor.name
                    result.evidence = list(dict.fromkeys(evidence + result.evidence))
                    result.artifacts = list(dict.fromkeys(artifacts + result.artifacts))
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
            return AgentResult(agent= self.descriptor.name, agent_execution_id=execution_id, status=AgentStatus.SUCCESS, summary=raw[:4000], result=raw,
                               evidence=evidence, artifacts=artifacts)
        except Exception as exc:
            return AgentResult(agent=self.descriptor.name, agent_execution_id=execution_id, status=AgentStatus.FAILURE, summary="Agent execution failed",
                               errors=[str(exc)[:500]])


class AgentRegistry:
    """Registry queried by capability; model identifiers stay in descriptors."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        name = agent.descriptor.name
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered")
        self._agents[name] = agent

    def get(self, name: str) -> BaseAgent:
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered")
        return self._agents[name]

    def list(self) -> list[AgentDescriptor]:
        return [agent.descriptor for agent in self._agents.values()]

    def capabilities(self) -> dict[str, list[str]]:
        return {d.name: [c.value for c in d.capabilities] for d in self.list()}

    def eligible(self, capabilities: Iterable[AgentCapability | str]) -> list[AgentDescriptor]:
        wanted = {c.value if isinstance(c, AgentCapability) else str(c) for c in capabilities}
        return [d for d in self.list() if d.available and wanted.issubset({c.value for c in d.capabilities})]


class AgentCapabilityPolicy:
    """Least-privilege policy shared by master, specialists, and tools."""

    def __init__(self, approvals: ApprovalManager | None = None) -> None:
        self.approvals = approvals or ApprovalManager()

    def can_use_tool(self, descriptor: AgentDescriptor, tool: str) -> bool:
        return tool in descriptor.allowed_tools

    def authorize(self, action: str, task_id: str, details: str = "",
                  approve: Callable[[str, str, str], bool] | None = None) -> bool:
        action = action.lower()
        if action == "read":
            return True
        if action == "network" or action == "delete":
            return False
        if action in {"edit", "execute", "git_commit"}:
            return bool(approve(task_id, action, details)) if approve else False
        return False


def build_default_agent_registry(model_registry: ModelRegistry,
                                 tools: dict[str, Callable[..., Any]] | None = None) -> AgentRegistry:
    """Create the initial specialists from configuration, not master logic."""
    definitions = [
        ("coding_agent", "coding specialist", "qwen-coder", [AgentCapability.CODING],
         ["list_directory", "read_file", "search_files", "find_files", "get_file_info", "git_status", "git_diff", "edit_file", "create_file", "create_python_script", "execute_command"]),
        ("vision_agent", "vision and diagram specialist", "qwen-vision", [AgentCapability.VISION], []),
        ("document_agent", "document analysis and creation specialist", "qwen-general", [AgentCapability.DOCUMENT], ["document_runner", "ocr_pdf", "create_document"]),
        ("lightweight_agent", "lightweight formatting specialist", "llama-small", [AgentCapability.LIGHTWEIGHT], []),
        ("general_agent", "general reasoning specialist", "qwen-general", [AgentCapability.GENERAL], []),
    ]
    registry = AgentRegistry()
    for name, role, provider_name, capabilities, allowed in definitions:
        try:
            provider = model_registry.get_provider(provider_name)
        except KeyError:
            # Synthetic/test registries may intentionally expose only a
            # subset.  Select an available provider by capability, otherwise
            # omit that specialist without affecting the legacy graph.
            wanted = {
                AgentCapability.CODING: "coding", AgentCapability.VISION: "vision",
                AgentCapability.DOCUMENT: "document_analysis", AgentCapability.LIGHTWEIGHT: "lightweight",
                AgentCapability.GENERAL: "general",
            }.get(capabilities[0], "general")
            matches = [p for p in model_registry.list_models()
                       if any(getattr(cap, "value", cap) == wanted for cap in p.capabilities)]
            providers = [model_registry.get_provider(cfg.name) for cfg in matches]
            if not providers:
                continue
            provider = providers[0]
            provider_name = provider.name
        registry.register(OllamaSpecialistAgent(
            AgentDescriptor(name=name, role=role, capabilities=capabilities,
                            provider_name=provider_name, allowed_tools=allowed,
                            constraints=["local-only", "bounded context"],
                            modality="vision" if AgentCapability.VISION in capabilities else
                                     "document" if AgentCapability.DOCUMENT in capabilities else "text",
                            description=role, permissions=["read"] + (["edit", "execute"] if AgentCapability.CODING in capabilities else [])),
            provider, tools,
        ))
    return registry


class MasterAgent:
    """Bounded PLAN → DELEGATE → REVIEW → VERIFY → FINAL workflow."""

    def __init__(self, registry: AgentRegistry, *, master_provider: ModelProvider | None = None,
                 planner: Callable[..., Awaitable[list[dict[str, Any]]]] | None = None,
                 verifier: Callable[[AgentResult], Awaitable[dict[str, Any]]] | None = None,
                 policy: AgentCapabilityPolicy | None = None, max_master_steps: int = 10,
                 max_subagent_calls: int = 8, max_depth: int = 2, total_timeout_seconds: float = 300.0,
                 trace: list[dict[str, Any]] | None = None,
                 progress_callback: Callable[[dict[str, Any]], None] | None = None,
                 capability_matcher: CapabilityMatcher | None = None) -> None:
        self.registry, self.master_provider, self.planner, self.verifier = registry, master_provider, planner, verifier
        self.policy = policy or AgentCapabilityPolicy()
        self.max_master_steps = min(10, max(1, max_master_steps))
        self.max_subagent_calls = min(8, max(1, max_subagent_calls))
        self.max_depth = max(0, max_depth)
        self.total_timeout_seconds = max(1.0, total_timeout_seconds)
        self.trace = trace if trace is not None else []
        self.progress_callback = progress_callback
        self.capability_matcher = capability_matcher or CapabilityMatcher([
            CapabilityProfile("document_creation", "create new documents and save DOCX, PDF, Markdown, or TXT artifacts", "document_agent", "document", ("docx", "pdf", "markdown", "md", "txt")),
            CapabilityProfile("document_analysis", "inspect PDFs, reports, OCR text, and extract findings", "document_agent", "document", ("json", "docx")),
            CapabilityProfile("p_and_id_analysis", "analyze P&ID process diagrams and engineering drawings", "vision_agent", "image", ("json",)),
            CapabilityProfile("code_debugging", "read, debug, edit source code and run tests", "coding_agent", "text", ("patch",)),
            CapabilityProfile("general_reasoning", "answer general questions and summarize information", "general_agent", "text", ("text",)),
        ])

    def _event(self, event: str, state: MasterTaskState, **meta: Any) -> None:
        self.trace.append({"event": event, "trace_id": state.trace_id,
                           "timestamp": time.time(), **meta})
        if self.progress_callback:
            # Progress contains phase/status metadata only; model reasoning and
            # raw prompts are intentionally never exposed to the terminal.
            self.progress_callback({"event": event, "trace_id": state.trace_id, **meta})

    def _capability_plan(self, request: str) -> list[dict[str, Any]]:
        """Master-owned capability selection used when model planning is unavailable."""
        text = request.lower()
        output_match = re.search(r"\b(docx|word document|microsoft word|pdf|markdown|md|txt)\b", text)
        output = output_match.group(1) if output_match else None
        modality = "image" if re.search(r"\b(image|p&id|diagram|visual)\b", text) else "document" if re.search(r"\b(pdf|document|report|ocr|markdown|md|txt)\b", text) else None
        code_intent = bool(re.search(r"\b(write|generate|create|implement|debug|fix)\b.*\b(code|python|function|script)\b", text))
        if code_intent:
            modality, output = "text", None
        ranked, _ = self.capability_matcher.rank(request, modality=modality, output=output, top_k=3)
        if ranked and ranked[0]["score"] >= 0.15:
            selected = ranked[0]
            return [{"agent": selected["agent"], "task": request, "capability": selected["capability"],
                     "selection_score": selected["score"], "success_criteria": ["structured evidence"]}]
        if re.search(r"\.(pdf|docx?)\b|\b(inspection|document|report|ocr)\b", text):
            return [{"agent": "document_agent", "task": request,
                     "success_criteria": ["structured document evidence"]}]
        if re.search(r"\.(png|jpe?g|webp|bmp|tiff?)\b|\b(image|p&id|diagram|visual)\b", text):
            return [{"agent": "vision_agent", "task": request,
                     "success_criteria": ["structured visual evidence"]}]
        if re.search(r"\.(py|js|ts|rs|go|java|c|cpp)\b|\b(code|coding|function|bug|pytest|test)\b", text):
            return [{"agent": "coding_agent", "task": request,
                     "success_criteria": ["source evidence and analysis"]}]
        if len(text.split()) <= 12:
            return [{"agent": "lightweight_agent", "task": request}]
        return [{"agent": "general_agent", "task": request}]

    async def delegate_to_agent(
        self,
        agent: str,
        task: str,
        context: dict[str, Any] | None = None,
        constraints: list[str] | None = None,
        expected_output: str = "structured result",
        success_criteria: list[str] | None = None,
        *,
        trace_id: str | None = None,
        depth: int = 0,
    ) -> AgentResult:
        """Delegate one bounded, structured task through the registry.

        The caller supplies a role name; provider/model selection remains in
        the registry.  Unknown roles and depth violations are returned as
        structured failures rather than escaping to arbitrary execution.
        """
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        if depth > self.max_depth:
            return AgentResult(agent_execution_id=execution_id, status=AgentStatus.BLOCKED,
                               summary="Delegation depth limit reached", errors=["max_depth"])
        try:
            specialist = self.registry.get(agent)
        except KeyError as exc:
            return AgentResult(agent_execution_id=execution_id, status=AgentStatus.BLOCKED,
                               summary="Unknown agent", errors=[str(exc)])
        request = AgentRequest(task=task, context=context or {}, constraints=constraints or [],
                               expected_output=expected_output, success_criteria=success_criteria or [],
                               trace_id=trace_id or uuid.uuid4().hex, depth=depth,
                               agent_execution_id=execution_id)
        result = await specialist.run(request)
        result.agent_execution_id = execution_id
        return result

    async def run(self, user_request: str, *, context: dict[str, Any] | None = None) -> MasterTaskState:
        nlp_result = NLPPreprocessor().process(user_request)
        state = MasterTaskState(user_request=user_request, preprocessing=nlp_result.to_dict())
        self._event("PREPROCESSING_STARTED", state, source="user")
        self._event("PREPROCESSING_COMPLETED", state,
                    normalization_count=nlp_result.metadata["normalization_count"],
                    entity_count=nlp_result.metadata["entity_count"],
                    enhancement_applied=nlp_result.metadata["enhancement_applied"],
                    processing_duration_ms=nlp_result.metadata["processing_duration_ms"])
        request_context = dict(context or {})
        planning_request = nlp_result.enhanced_prompt
        task_spec: dict[str, Any] = {"operation": "analyze", "original_request": user_request}
        request_context.setdefault("nlp", {
            "original_text": nlp_result.original_text,
            "normalized_text": nlp_result.normalized_text,
            "enhanced_prompt": nlp_result.enhanced_prompt,
            "entities": nlp_result.entities,
            "keywords": nlp_result.keywords,
        })
        request_context.setdefault("task_spec", task_spec)
        path_match = re.search(r"(?:^|\s)([^\s]+\.(?:pdf|docx?|png|jpe?g|webp|bmp|tiff?|py|js|ts|rs|go|java|c|cpp))\b", planning_request, re.I)
        if path_match:
            candidate = path_match.group(1).strip('`\"\'.,')
            if candidate.lower().endswith((".pdf", ".doc", ".docx")):
                request_context.setdefault("input_path", candidate)
            elif candidate.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
                request_context.setdefault("image_paths", [candidate])
            else:
                request_context.setdefault("path", candidate)
        self._event("MASTER_PLAN", state)
        if self.planner:
            try:
                state.master_plan = await self.planner(planning_request, self.registry.capabilities())
            except Exception as exc:
                # Capability planning below is the bounded local recovery
                # path; a malformed/empty model plan is a warning, not a task
                # failure when a valid specialist can still be selected.
                self._event("MASTER_PLAN_WARNING", state, reason=str(exc)[:300])
        elif self.master_provider is not None:
            try:
                prompt = ("Understand the user's objective first, then plan it. Return one JSON object with "
                          "task_spec {operation, topic, content_requirements, artifact_format, modality} and "
                          "plan (a list of {agent, task, success_criteria}); never mention model IDs.\n"
                          f"Capabilities: {json.dumps(self.registry.capabilities())}\nRequest: {planning_request[:4000]}")
                response = await asyncio.wait_for(self.master_provider.generate(prompt), self.total_timeout_seconds)
                parsed = json.loads(response.content)
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("task_spec"), dict):
                        task_spec.update(parsed["task_spec"])
                        request_context["task_spec"] = task_spec
                    parsed = parsed.get("plan", [])
                if isinstance(parsed, list):
                    state.master_plan = [item for item in parsed if isinstance(item, dict)]
            except Exception as exc:
                self._event("MASTER_PLAN_WARNING", state, reason=str(exc)[:300])
        planned = self._capability_plan(planning_request)
        expected = planned[0]["agent"]
        if planned and "selection_score" in planned[0]:
            self._event("CAPABILITY_MATCH", state, capability=planned[0].get("capability"),
                        selected_agent=expected, similarity=planned[0].get("selection_score"),
                        top_k=3, embedding_model="local-semantic-hash")
        registered = {d.name for d in self.registry.list()}
        if (not state.master_plan or
                any(item.get("agent") not in registered for item in state.master_plan) or
                (expected in registered and expected in {"document_agent", "vision_agent", "coding_agent"}
                 and state.master_plan[0].get("agent") != expected)):
            state.master_plan = self._capability_plan(user_request)
        # Keep all planned retries within the capability required by the
        # original request. A document failure must never fall through to an
        # unrelated coding item supplied by a planner.
        scoped_request = bool(re.search(
            r"\b(create|write|generate)\b.*\b(document|docx|pdf|markdown|md|txt|report)\b|"
            r"\b(image|p&id|diagram|visual)\b|"
            r"\b(fix|debug|modify|implement|write|create)\b.*\b(code|python|function|script|pytest|test)\b",
            planning_request, re.I))
        if scoped_request and expected in {"document_agent", "vision_agent", "coding_agent"}:
            compatible = [item for item in state.master_plan if item.get("agent") == expected]
            state.master_plan = compatible[: self.max_master_steps] or self._capability_plan(planning_request)
        self._event("CAPABILITY_DISCOVERY", state, capabilities=self.registry.capabilities())
        calls = 0
        for step, item in enumerate(state.master_plan[: self.max_master_steps]):
            if calls >= self.max_subagent_calls:
                state.errors.append("subagent call limit reached")
                break
            name = str(item.get("agent", ""))
            try:
                agent = self.registry.get(name)
            except KeyError as exc:
                state.errors.append(str(exc)); continue
            if step >= self.max_depth + 1:
                state.errors.append("delegation depth limit reached"); break
            execution_id = f"exec_{uuid.uuid4().hex[:12]}"
            request = AgentRequest(task=str(item.get("task", planning_request)), context=request_context,
                                   constraints=_coerce_list_of_strings(item.get("constraints", [])),
                                   expected_output=str(item.get("expected_output", "structured result")),
                                   success_criteria=_coerce_list_of_strings(item.get("success_criteria", [])),
                                   trace_id=state.trace_id, depth=step,
                                   agent_execution_id=execution_id)
            state.subtasks.append(request); state.active_subtask = step
            self._event("MASTER_DELEGATE", state, agent=agent.descriptor.name, agent_execution_id=execution_id)
            self._event("AGENT_START", state, agent=agent.descriptor.name, role=agent.descriptor.role, agent_execution_id=execution_id)
            calls += 1
            result = await agent.run(request)
            result.agent_execution_id = execution_id
            state.agent_results.append(result)
            state.artifacts.extend(result.artifacts)
            self._event("AGENT_COMPLETE", state, agent=agent.descriptor.name, status=result.status.value, agent_execution_id=execution_id)
            self._event("MASTER_REVIEW", state, status=result.status.value, agent_execution_id=execution_id)
            if result.status in {AgentStatus.FAILURE, AgentStatus.BLOCKED}:
                state.errors.extend(result.errors or [result.summary])
                artifact = next((x for x in ("pdf", "docx", "markdown", "md", "txt")
                                 if re.search(rf"\b{x}\b", planning_request, re.I)), "")
                self._event("MASTER_REPLAN", state, reason=result.summary[:300],
                            failed_agent=agent.descriptor.name,
                            required_capability=expected, artifact=artifact,
                            compatible_candidates=[d.name for d in self.registry.list()
                                                   if d.name == expected])
                continue
            if self.verifier:
                state.verification = await self.verifier(result)
                self._event("VERIFICATION", state, status=state.verification.get("status", "unknown"))
            state.final_answer = result.summary or str(result.result or "")
            break
        if not state.final_answer:
            state.final_answer = "Unable to complete the requested task with the available specialists."
        self._event("FINAL", state, status="failure" if state.errors and not state.agent_results else "success")
        return state
