# AEGIS Backend Handoff

## 1. Stable architecture

The backend baseline is the deterministic AEGIS path:

```text
CLI/API
  → NLP normalization
  → MasterAgent capability routing
  → LangGraph/state-machine orchestration
  → specialist agent
  → PolicyEngine and typed tools
  → sandboxed execution or document workflow
  → deterministic verification
  → checkpoint/audit persistence
```

Explicit workflow integrations additionally use:

```text
Principal → RBAC/policy → ExecutionPlan → preflight validation
  → SovereignExecutor → risk approval → route → execute → verify → AuditChain
```

The stable baseline includes:

- local Ollama model execution;
- capability-based specialist routing;
- coding-agent workflow;
- sandboxed command/code execution;
- retry and failure recovery;
- SQLite checkpoint/resume behavior;
- vision and multimodal paths;
- document generation;
- verification and evidence flow;
- PolicyEngine authorization boundaries;
- human approval gates;
- circuit breakers and bounded execution;
- audit/Merkle integrity records;
- routing telemetry;
- golden/regression tests;
- security demonstration scripts.
- typed workflow contracts and versioned registry;
- local hybrid retrieval with source citations;
- scoped SQLite memory with TTL;
- retryable/cancellable background jobs;
- evaluation records and API summaries;
- visual workflow export and Kubernetes manifest planning as explicit extensions.

## 2. Installation and run

```bash
bash scripts/setup.sh
source .venv/bin/activate
# Pull the model tags listed in config/models.yaml with Ollama.
.venv/bin/python cli.py
```

The normal application entry point is `Orchestrator.run_master`. Runtime state,
checkpoints, and telemetry are stored under the ignored `workspace/.aegis/`
directory. Generated artifacts are stored under `workspace/outputs/`.

## 3. Environment variables

- `AEGIS_ROUTING_MODE=shadow` is the safe default.
- `AEGIS_ROUTING_MODE=adaptive` is experimental and must not be enabled for the
  backend baseline.
- `AEGIS_ROUTING_EVAL_TIMEOUT_SECONDS` bounds actual routing experiments.
- `VISION_TIMEOUT_SECONDS` and `DOCUMENT_TIMEOUT_SECONDS` bound specialist work.

## 4. Interfaces

- CLI: `cli.py`
- API: `app/api/main.py`
- Orchestrator: `runtime/orchestrator.py`
- Specialist registry: `runtime/agents.py`
- Tool authorization: `runtime/tool_policy.py`
- Checkpoints: `storage/graph_state.py`
- Telemetry: `storage/telemetry.py`
- Typed plans and validation: `aegis/contracts.py`
- Governance and audit chain: `aegis/governance.py`
- Retrieval and memory: `aegis/retrieval.py`, `aegis/memory.py`
- Sovereign typed-plan seam: `aegis/sovereign.py`

## 5. Model requirements

The default local profiles are configured in `config/models.yaml`:

- `qwen-general` for master/general/document reasoning;
- `qwen-coder` for coding;
- `qwen-vision` for vision and scanned-document work;
- `llama-small` for lightweight capability profiles.

## 6. Security contract

Every side-effecting action remains subject to PolicyEngine. The adaptive
router cannot grant filesystem access, network access, command authorization,
destructive permissions, or approval authority. Official document generation,
outside-workspace writes, forbidden network access, timeout, and oversized
output paths remain bounded or approval-gated.

## 7. Validation commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/evaluate_golden.py --repeats 3
.venv/bin/python scripts/security_demo.py
.venv/bin/python cli.py
```

The routing contract evaluator is diagnostic only:

```bash
.venv/bin/python scripts/evaluate_routing.py --repeats 3
```

Real specialist comparison is explicitly opt-in and may be slow:

```bash
.venv/bin/python scripts/evaluate_routing.py --repeats 3 --actual
```

## 8. Known limitations

The deterministic routing baseline is the supported backend contract. Real
adaptive comparison still requires a larger matched dataset of executed local
candidate workflows. Contract compatibility is not model-quality evidence.

## 9. RL/adaptive routing status

LinUCB, training observations, policy persistence, shadow decisions, promotion
experiments, and adaptive policy learning are experimental. They are not part
of the backend baseline. Backend integrations must use deterministic routing
and leave `AEGIS_ROUTING_MODE` unset or set to `shadow`.

The adaptive layer must pass the security-first quality gate before any future
promotion. It must never weaken PolicyEngine decisions.

The SIH2026 presentation output uses the personal
`artifact-template-sih2026-aegis-presentation` template and is separate from
the backend runtime.

## 10. Handoff revision

The commit SHA and branch are supplied with the handoff. Do not depend on
adaptive routing APIs for baseline backend integration.
