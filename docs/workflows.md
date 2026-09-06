# AEGIS Workflows — Current Status

All normal CLI requests and API tasks enter the same MasterAgent workflow.
NLP preprocessing preserves the original request and supplies normalized
context. Master capability discovery uses registry metadata and semantic
matching, then delegates to one least-privileged specialist.

```text
request → NLP → MASTER_PLAN → CAPABILITY_DISCOVERY → DELEGATE
→ specialist/tool execution → structured AgentResult → REVIEW
→ VERIFY or bounded REPLAN → FINAL + trace/artifacts
```

Reads are automatic. Edits and approved test commands require terminal
approval; workspace escapes, network access, deletes, arbitrary shell, and Git
mutation are denied. Specialist failures become structured failures or
capability-compatible retries; there is no universal coding fallback.

---

## 1. Master Workflow (authoritative)

When a user initiates an interaction via `cli.py` or the API, the orchestrator executes the Master pipeline above. The older classifier/router graph is retained only as compatibility infrastructure and is not the normal execution path.

```text
User/OCR → NLP → MasterAgent → AgentRegistry → specialist
→ Policy → tools/pipelines → AgentResult → Master review → final
```

---

## 2. Multi-Step Workflows (Roadmap & Progressive Integration)

### Workflow 1: Flagship Inspection & Approval Note (Phases 3-5)
```text
Inspection Report (PDF)
         │
         ▼
[Document Pipeline] ──▶ Detect Native vs Scanned
         │
         ▼
[OCR / PP-StructureV3] ──▶ Extract structured findings, tables, defect logs
         │
         ▼
[Reasoning Model] ──▶ Synthesize technical recommendation & risk level
         │
         ▼
[HITL Approval] ──▶ Operator reviews findings & draft approval note
         │
         ▼
[Artifact Generator] ──▶ Generates signed inspection_approval_note.docx
         │
         ▼
[Artifact Store] ──▶ Records metadata, SHA256 checksum, and file size
```

### Workflow 2: Code Generation with Sandbox Verification (Phases 2-5)
```text
Coding Request
      │
      ▼
[Coding Model] ──▶ Generates Python code & unit tests
      │
      ▼
[Docker Sandbox] ──▶ Executes in ephemeral container (network disabled)
      │
      ├── If tests fail ──▶ Feed error back to Coding Model for self-repair
      │
      └── If tests pass ──▶ Return verified code & test execution proof
```
