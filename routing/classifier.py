"""Task classification — deterministic analysis of user requests.

Uses keyword / heuristic patterns, file path extraction, and modality inspection
to classify tasks without burning GPU cycles.
"""

from __future__ import annotations

import ast
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from runtime.regex_safety import bounded_text


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    GENERAL = "general"
    CODING = "coding"
    DEBUGGING = "debugging"
    DOCUMENT_ANALYSIS = "document_analysis"
    SUMMARIZATION = "summarization"
    CALCULATION = "calculation"
    MULTIMODAL = "multimodal"
    IMAGE_ANALYSIS = "image_analysis"
    ENGINEERING_DOCUMENT = "engineering_document"
    ARTIFACT_GENERATION = "artifact_generation"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    MULTIMODAL = "multimodal"


class Complexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ExecutionMode(str, Enum):
    """Routing mode: direct deterministic tool, model, or model with tools."""
    DIRECT_TOOL = "direct_tool"
    MODEL = "model"
    MODEL_WITH_TOOLS = "model_with_tools"


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """Structured task object produced by the classifier."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    user_request: str
    modality: Modality = Modality.TEXT
    task_type: TaskType = TaskType.GENERAL
    complexity: Complexity = Complexity.SIMPLE
    execution_mode: ExecutionMode = ExecutionMode.MODEL
    direct_tool_name: str | None = None
    direct_tool_args: dict[str, Any] = Field(default_factory=dict)
    requires_vision: bool = False
    requires_code: bool = False
    requires_documents: bool = False
    requires_tools: list[str] = Field(default_factory=list)
    requires_human_approval: bool = False
    attached_files: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_serializable_dict(self) -> dict[str, Any]:
        """Convert to primitive-only dictionary safe for LangGraph serialization."""
        data = self.model_dump(mode="json")
        # Ensure enums are converted to strings
        data["modality"] = self.modality.value
        data["task_type"] = self.task_type.value
        data["complexity"] = self.complexity.value
        data["execution_mode"] = self.execution_mode.value
        if isinstance(data.get("created_at"), datetime):
            data["created_at"] = data["created_at"].isoformat()
        return data


# ---------------------------------------------------------------------------
# File extension sets
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS: set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".svg",
}
_DOCUMENT_EXTENSIONS: set[str] = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".txt", ".csv", ".json", ".md", ".rtf",
}
_CODE_EXTENSIONS: set[str] = {
    ".py", ".sh", ".c", ".cpp", ".h", ".rs", ".go", ".js", ".ts",
    ".java", ".sql", ".html", ".css",
}

ALL_KNOWN_EXTENSIONS: set[str] = (
    _IMAGE_EXTENSIONS | _DOCUMENT_EXTENSIONS | _CODE_EXTENSIONS
)


# ---------------------------------------------------------------------------
# Keyword / regex patterns
# ---------------------------------------------------------------------------

_CODING_PATTERNS: list[str] = [
    r"\b(write|create|build|implement|code|program|script|function|class)\b",
    r"\b(?:write|create|build|implement|code|program|script|function|class|debug|fix)\b.*\b(python|javascript|java|c\+\+|rust|go|html|css|sql|bash|shell)\b|\b(python|javascript|java|c\+\+|rust|go|html|css|sql|bash|shell)\b.*\b(?:write|create|build|implement|code|program|script|function|class|debug|fix)\b",
    r"\b(algorithm|data structure|api|endpoint|server|database|framework)\b",
    r"\b(binary tree|linked list|sorting|binary search|recursion)\b",
    r"\b(refactor|optimise|optimize|port|migrate|convert)\b.*\b(code|program|script)\b",
]

_SOURCE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
    ".yaml", ".yml", ".json", ".toml", ".md", ".rs", ".go",
}

_WORKSPACE_PATTERNS: list[str] = [
    r"\b(?:list|show)\b.*\b(?:files?|directory|folder|structure)\b",
    r"(?<!binary\s)\btree\b|\brecursive(?:ly)?\s+(?:file|directory)|\b(?:directory|file)\s+structure\b",
    r"\bwhat files are here\b",
    r"\bfind\b.*\b(?:files?|python files?)\b",
    r"\bsearch\b.*\b(?:for|in)\b",
    r"\bopen\s+[\w./-]+\b",
    r"\bgit\s+(?:diff|status)\b",
    r"\bfind where\b.*\b(?:routed|defined|used|handled)\b",
    r"\bsearch\s+(?:for\s+)?\w+\b",
]


def _has_source_action(req: str) -> bool:
    """Check if request combines a source-file action with a source file."""
    has_action = bool(re.search(
        r"\b(inspect|search|open|read|show|find|view|cat|check|git|"
        r"edit|modify|change|update|add|remove|delete|write|create)\b",
        req,
        re.I,
    ))

    if not has_action:
        return False

    for token in req.split():
        cleaned = token.strip(",;:\"'()[]{}<>`").lower()
        if any(cleaned.endswith(ext) for ext in _SOURCE_EXTENSIONS):
            return True

    return False

def _has_source_mutation(req: str) -> bool:
    """Check whether a source-file request intends to modify the workspace."""
    return bool(re.search(
        r"\b(edit|modify|change|update|add|remove|delete|write|create)\b",
        req,
        re.I,
    )) and _has_source_action(req)


def _is_workspace_intent(req: str) -> bool:
    """Recognize explicit local-project workspace requests."""
    if _has_source_action(req):
        return True
    return _matches_any(req, _WORKSPACE_PATTERNS)

_DEBUGGING_PATTERNS: list[str] = [
    r"\b(debug|fix|error|bug|issue|traceback|exception|crash|fail(ed|ure)?)\b",
    r"\b(not working|broken|wrong output|unexpected)\b",
    r"\bstack\s?trace\b",
]

_VISION_PATTERNS: list[str] = [
    r"\b(image|photo|picture|screenshot|diagram|drawing|sketch|scan(ned)?)\b",
    r"\b(look at|see|visual|what is in|describe this|explain this)\b.*\b(image|photo|picture|diagram)\b",
]

_DOCUMENT_PATTERNS: list[str] = [
    r"\b(document|report|pdf|docx|xlsx|pptx|spreadsheet|presentation)\b",
    r"\b(read|analyze|extract|parse|summarize|review)\b.*\b(file|document|report|pdf)\b",
    r"\b(inspection|approval|office note|board|memo|manual|certificate)\b",
]

_SUMMARIZATION_PATTERNS: list[str] = [
    r"\b(summarize|summary|brief|overview|key points|tldr|gist)\b",
]

_ARTIFACT_PATTERNS: list[str] = [
    r"\b(generate|create|make|produce|draft|prepare|write)\b.*\b(document|report|note|file|docx|pdf|xlsx|pptx)\b",
    r"\b(approval note|office note|memo|letter|certificate)\b",
]

_ENGINEERING_PATTERNS: list[str] = [
    r"\b(p&id|pid|piping|instrumentation|engineering drawing)\b",
    r"\b(valve|pump|tank|pipe|flow|process)\b.*\b(diagram|drawing|schematic)\b",
]


# ---------------------------------------------------------------------------
# File path extraction helper
# ---------------------------------------------------------------------------

def extract_file_paths(text: str) -> list[str]:
    """Extract file paths from user prompt.

    Handles:
    - Quoted paths: "path/to/file.ext" or 'path/to/file.ext'
    - Absolute paths: /path/to/image.jpg
    - Relative paths: ./image.png, workspace/sample.pdf
    - Filenames with extensions: image.jpg, report.pdf
    """
    found: list[str] = []

    # 1. Quoted paths
    for m in re.finditer(r'["\']([^"\']+\.[a-zA-Z0-9_-]+)["\']', text):
        candidate = m.group(1).strip()
        if candidate and candidate not in found:
            found.append(candidate)

    # 2. Unquoted paths or filenames with known extensions
    # Split by whitespace, strip punctuation from ends
    tokens = text.split()
    for token in tokens:
        cleaned = token.strip(",;:\"'()[]{}<>`")
        if not cleaned:
            continue
        p = Path(cleaned)
        # Check if has a recognized extension
        if p.suffix.lower() in ALL_KNOWN_EXTENSIONS:
            if cleaned not in found:
                found.append(cleaned)
        # Check if starts with / or ./ or ../ and exists or looks like path
        elif cleaned.startswith(("/", "./", "../")) and ("/" in cleaned):
            if cleaned not in found:
                found.append(cleaned)

    return found


# ---------------------------------------------------------------------------
# Direct Tool Matchers
# ---------------------------------------------------------------------------

def _try_match_direct_calculator(req: str) -> tuple[bool, str]:
    """Check if request is a direct math/calculator calculation.

    Returns (is_direct, expression).
    """
    clean = req.strip()

    # If it contains words asking for explanation/theory, not a direct calculation
    if re.search(r"\b(explain|teach|how to|why|history|concept|derive|proof)\b", clean, re.I):
        return False, ""

    # Check for prefix: "calculate ...", "compute ...", "what is ...", "eval ..."
    m = re.match(r"^(?:calculate|compute|what\s+is|evaluate|eval)\s+(.+?)[\?\.\!]*$", clean, re.I)
    if m:
        expr = m.group(1).strip()
    else:
        # Check if it's pure arithmetic: e.g. "234 * 918" or "sqrt(144) + 2"
        expr = clean.rstrip("?.!")

    # Validate if it looks like a math expression
    # Must contain at least one math operator or known math function
    has_operator = bool(re.search(r"[\+\-\*\/\%\^]|sqrt|log|sin|cos|tan|abs|pow", expr, re.I))
    # Must contain at least one digit
    has_digit = bool(re.search(r"\d", expr))

    if not (has_operator and has_digit):
        return False, ""

    # Try AST parse to ensure it's valid numeric math syntax
    try:
        parsed = ast.parse(expr, mode="eval")
        # Ensure it contains no statements, lambdas, or dangerous nodes
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
                if isinstance(node, ast.Call):
                    # Only allow known math functions
                    if isinstance(node.func, ast.Name):
                        if node.func.id not in (
                            "sqrt", "log", "log10", "log2", "sin", "cos", "tan",
                            "abs", "round", "min", "max", "ceil", "floor", "factorial", "pow"
                        ):
                            return False, ""
                    else:
                        return False, ""
            elif isinstance(node, (ast.Lambda, ast.Yield, ast.ListComp, ast.DictComp)):
                return False, ""
        return True, expr
    except Exception:
        return False, ""


def _try_match_direct_file_tool(req: str) -> tuple[bool, str, dict[str, Any]]:
    """Check if request is a direct file operation (list, read, create dir).

    Returns (is_direct, tool_name, tool_args).
    """
    clean = req.strip()

    # Explicit local-agent capabilities.  These do not change ordinary image
    # routing: OCR is selected only when the user explicitly requests it.
    ocr_paths = extract_file_paths(clean)
    convert_match = re.search(
        r"\bconvert\s+(?:the\s+)?(?:pdf|file)?\s*['\"]?([^'\"]+\.pdf)['\"]?\s+to\s+(?:images?|png)",
        clean,
        re.I,
    )
    if convert_match:
        return True, "convert_pdf_to_images", {"path": convert_match.group(1).strip()}
    if re.search(r"\b(?:ocr|optical\s+character\s+recognition)\b", clean, re.I):
        unsupported_ocr_path = next(
            (path for path in ocr_paths if Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg"}),
            None,
        )
        if unsupported_ocr_path:
            if Path(unsupported_ocr_path).suffix.lower() == ".pdf":
                return True, "ocr_pdf", {"path": unsupported_ocr_path}
            return True, "safe_extract_ocr", {"path": unsupported_ocr_path}
    unsupported_document_path = next(
        (
            path for path in ocr_paths
            if Path(path).suffix.lower() in {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}
        ),
        None,
    )
    if unsupported_document_path and re.search(
        r"\b(?:explain|analy[sz]e|analysis|summari[sz]e|review|interpret)\b",
        clean,
        re.I,
    ):
        return True, "safe_read_file", {"path": unsupported_document_path}
    m = re.match(r"^(?:inspect\s+(?:path\s+)?)['\"]?(.+?)['\"]?$", clean, re.I)
    if m:
        return True, "inspect_path", {"path": m.group(1).strip()}
    m = re.match(r"^(?:ocr|extract\s+ocr\s+(?:from\s+)?)['\"]?(.+?)['\"]?$", clean, re.I)
    if m:
        return True, "safe_extract_ocr", {"path": m.group(1).strip()}
    m = re.match(r"^(?:find|search)\s+(?:files?\s+)?(?:in\s+)?['\"]?([^'\"]+?)['\"]?(?:\s+(?:matching|pattern)\s+['\"]?([^'\"]+)['\"]?)?$", clean, re.I)
    if m:
        return True, "find_files", {"path": m.group(1).strip(), "pattern": (m.group(2) or "*").strip()}

    # 1. Direct file listing
    # Examples: "list all files in the directory", "list files", "ls", "show files"
    if re.search(r"^(list\s+(all\s+)?files|show\s+files|dir\b|ls\b|list\s+directory)", clean, re.I):
        # If user explicitly says "in this directory" / "in the directory" / "in current directory"
        if re.search(r"\b(this|the|current)\s+directory\b", clean, re.I):
            target_dir = "."
        else:
            m = re.search(r"(?:in|of)\s+([^\s]+)", clean, re.I)
            if m:
                matched_dir = m.group(1).strip("'\";,.")
                if matched_dir.lower() in ("this", "the", "current", "here", "dir", "directory"):
                    target_dir = "."
                else:
                    target_dir = matched_dir
            else:
                target_dir = "."
        return True, "list_files", {"directory": target_dir}

    # 2. Direct read file
    # Only if NOT asking to summarize, analyze, or explain the file
    if not re.search(r"\b(summarize|summary|analyze|analysis|explain|review|debug|fix|rewrite)\b", clean, re.I):
        m = re.search(r"^(?:read\s+file|cat\b|view\s+file|show\s+contents?\s+of)\s+([^\s]+)", clean, re.I)
        if m:
            file_path = m.group(1).strip("'\";,")
            return True, "read_file", {"file_path": file_path}

    # 3. Direct create directory
    m = re.search(r"^(?:create\s+directory|mkdir)\s+([^\s]+)", clean, re.I)
    if m:
        dir_path = m.group(1).strip("'\";,")
        return True, "create_directory", {"directory": dir_path}

    return False, "", {}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TaskClassifier:
    """Deterministic task classifier.

    Analyses user text + attached file extensions to produce a ``Task``.
    No LLM calls are made — this is fast and predictable.
    """

    def classify(
        self,
        user_request: str,
        attached_files: list[str] | None = None,
        previous_task_type: str | None = None,   # NEW
    ) -> Task:
        attached_files = list(attached_files or [])
        req = bounded_text(user_request).strip()

        extracted_paths = extract_file_paths(req)
        for ep in extracted_paths:
            if ep not in attached_files:
                attached_files.append(ep)

        modality = self._detect_modality(req.lower(), attached_files)

        # Keep terse shell aliases deterministic, but give natural-language project
        # inspection requests to the bounded coder/tool loop.
        if req.lower() not in {"ls", "dir"} and _is_workspace_intent(req.lower()):
            mutation = _has_source_mutation(req.lower())
            return Task(
                user_request=user_request,
                modality=Modality.TEXT,
                task_type=TaskType.CODING,
                complexity=Complexity.SIMPLE,
                execution_mode=ExecutionMode.MODEL_WITH_TOOLS,
                requires_code=True,
                requires_tools=(
                    ["workspace_read", "edit_file"]
                    if mutation
                    else ["workspace_read"]
                ),
                requires_human_approval=mutation,
                attached_files=attached_files,
            )

        is_calc, expr = _try_match_direct_calculator(req)
        if is_calc:
            return Task(
                user_request=user_request, modality=Modality.TEXT,
                task_type=TaskType.CALCULATION, complexity=Complexity.SIMPLE,
                execution_mode=ExecutionMode.DIRECT_TOOL, direct_tool_name="calculator",
                direct_tool_args={"expression": expr}, requires_tools=["calculator"],
                attached_files=attached_files,
            )

        is_file_tool, tool_name, tool_args = _try_match_direct_file_tool(req)
        if is_file_tool and tool_name == "ocr_pdf":
            is_file_tool = False
        if is_file_tool:
            return Task(
                user_request=user_request,
                modality=Modality.TEXT if tool_name == "list_files" else Modality.DOCUMENT,
                task_type=TaskType.GENERAL if tool_name == "list_files" else TaskType.DOCUMENT_ANALYSIS,
                complexity=Complexity.SIMPLE, execution_mode=ExecutionMode.DIRECT_TOOL,
                direct_tool_name=tool_name, direct_tool_args=tool_args,
                requires_tools=[tool_name], attached_files=attached_files,
            )

        task_type = self._detect_task_type(req.lower(), modality)

        # NEW: continuity fallback. Only overrides GENERAL, only when the message
        # is short and carries no strong signal of its own, and only pulls toward
        # CODING/DEBUGGING specifically — never toward DOCUMENT_ANALYSIS or other
        # types, since those are less likely to appear as bare data fragments.
        if task_type == TaskType.GENERAL and previous_task_type in (
            TaskType.CODING.value, TaskType.DEBUGGING.value
        ):
            word_count = len(req.split())
            has_own_signal = _matches_any(
                req.lower(),
                _CODING_PATTERNS + _DEBUGGING_PATTERNS + _DOCUMENT_PATTERNS
                + _ARTIFACT_PATTERNS + _SUMMARIZATION_PATTERNS,
            )
            looks_like_data_fragment = bool(
                re.search(r"[\{\[]|^\s*['\"]?\w+['\"]?\s*[:=]", req)
            )
            if word_count <= 12 and not has_own_signal and (
                looks_like_data_fragment or word_count <= 4
            ):
                task_type = TaskType(previous_task_type)

        complexity = self._estimate_complexity(req.lower(), task_type, attached_files)

        requires_vision = (modality in (Modality.IMAGE, Modality.MULTIMODAL))
        requires_code = (task_type in (TaskType.CODING, TaskType.DEBUGGING))
        requires_documents = (
            modality == Modality.DOCUMENT or task_type == TaskType.DOCUMENT_ANALYSIS
        )
        required_tools = self._detect_required_tools(task_type, modality)
        if (
            re.search(r"\b(?:ocr|optical\s+character\s+recognition)\b", req, re.I)
            and any(Path(path).suffix.lower() == ".pdf" for path in attached_files)
        ):
            required_tools.append("ocr_pdf")

        if requires_code and re.search(r"\b(test|verify|execute|run|sandbox)\b", req, re.I):
            execution_mode = ExecutionMode.MODEL_WITH_TOOLS
        elif requires_documents and required_tools:
            execution_mode = ExecutionMode.MODEL_WITH_TOOLS
        else:
            execution_mode = ExecutionMode.MODEL

        return Task(
            user_request=user_request, modality=modality, task_type=task_type,
            complexity=complexity, execution_mode=execution_mode,
            requires_vision=requires_vision, requires_code=requires_code,
            requires_documents=requires_documents, requires_tools=required_tools,
            requires_human_approval=(
                task_type == TaskType.ARTIFACT_GENERATION
                or _matches_any(req.lower(), _ARTIFACT_PATTERNS)
            ),
            attached_files=attached_files,
        )
       

    # -- internal helpers -----------------------------------------------

    def _detect_modality(self, req: str, files: list[str]) -> Modality:
        has_images = any(
            Path(f).suffix.lower() in _IMAGE_EXTENSIONS for f in files
        )
        has_docs = any(
            Path(f).suffix.lower() in _DOCUMENT_EXTENSIONS for f in files
        )
        if has_images and has_docs:
            return Modality.MULTIMODAL
        if has_images:
            return Modality.IMAGE
        if has_docs:
            return Modality.DOCUMENT
        # Fall back to text cues (no files attached)
        if _matches_any(req, _VISION_PATTERNS):
            return Modality.IMAGE
        if _matches_any(req, _DOCUMENT_PATTERNS):
            return Modality.DOCUMENT
        return Modality.TEXT

    def _detect_task_type(self, req: str, modality: Modality) -> TaskType:
        # If image is detected, it is an image analysis or multimodal task
        if modality == Modality.IMAGE:
            return TaskType.IMAGE_ANALYSIS
        if modality == Modality.MULTIMODAL:
            return TaskType.MULTIMODAL

        # If user asks to explain, describe, or define a concept without coding action verbs,
        # it is a GENERAL conceptual task.
        is_concept_query = bool(re.search(r"^(explain|what\s+is|describe|define|tell\s+me\s+about)\b", req, re.I))
        has_code_action = bool(re.search(
            r"\b(write|create|edit|modify|change|update|add|remove|delete|"
            r"code|implement|program|script|build|debug|fix|refactor)\b",
            req, re.I
        ))
        if is_concept_query and not has_code_action:
            return TaskType.GENERAL

        # More specific patterns first
        if _matches_any(req, _ENGINEERING_PATTERNS):
            return TaskType.ENGINEERING_DOCUMENT
        if _matches_any(req, _DEBUGGING_PATTERNS):
            return TaskType.DEBUGGING
        if _matches_any(req, _CODING_PATTERNS):
            return TaskType.CODING
        if _matches_any(req, _ARTIFACT_PATTERNS):
            return TaskType.ARTIFACT_GENERATION
        if _matches_any(req, _SUMMARIZATION_PATTERNS):
            return TaskType.SUMMARIZATION
        if modality == Modality.DOCUMENT:
            return TaskType.DOCUMENT_ANALYSIS
        return TaskType.GENERAL

    @staticmethod
    def _estimate_complexity(
        req: str,
        task_type: TaskType,
        files: list[str],
    ) -> Complexity:
        words = len(req.split())
        has_files = bool(files)
        complex_types = {
            TaskType.ENGINEERING_DOCUMENT,
            TaskType.ARTIFACT_GENERATION,
            TaskType.MULTIMODAL,
        }
        if task_type in complex_types or (has_files and words > 50):
            return Complexity.COMPLEX
        moderate_types = {
            TaskType.CODING,
            TaskType.DEBUGGING,
            TaskType.DOCUMENT_ANALYSIS,
        }
        if task_type in moderate_types or words > 20:
            return Complexity.MODERATE
        return Complexity.SIMPLE

    @staticmethod
    def _detect_required_tools(
        task_type: TaskType,
        modality: Modality,
    ) -> list[str]:
        tools: list[str] = []
        if task_type == TaskType.CALCULATION:
            tools.append("calculator")
        if task_type in (TaskType.CODING, TaskType.DEBUGGING):
            tools.append("python_executor")
        if modality == Modality.DOCUMENT:
            tools.append("read_file")
        if modality in (Modality.IMAGE, Modality.MULTIMODAL):
            tools.append("vision")
        if task_type == TaskType.ARTIFACT_GENERATION:
            tools.append("artifact_generator")
        return tools


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _matches_any(text: str, patterns: list[str]) -> bool:
    """Return True if *text* matches any of the compiled regex *patterns*."""
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)
