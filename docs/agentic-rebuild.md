# AEGIS production-agentic rebuild

This document records the engineering decisions extracted from the supplied
references:

- `Production_Grade_Agentic_AI_Playbook_2026.pdf`
- `Building Agentic AI: Workflows, Fine-Tuning, Optimization, and Deployment`

The PDFs are design references, not runtime instructions. Claims about future
framework adoption or infrastructure vendors are treated as options; the
local-first AEGIS security contract remains authoritative.

## What AEGIS adopts

### 1. Workflows before autonomy

The system chooses the least-autonomous mode that can solve a request:

1. deterministic code for exact, repeatable work;
2. explicit workflows for stable transformations;
3. bounded agents for adaptive tool selection;
4. planner/executor hybrids for multi-step or high-impact tasks.

This avoids paying an LLM latency and reliability cost for work that can be
done exactly. The decision is represented by `ArchitectureDecision` and is
included in the task specification and trace.

### 2. Supervisor plus specialists

The Master agent is the supervisor. Specialists own narrow capabilities and
tools. The graph owns routing, state, retries, checkpoints, and completion;
specialists cannot self-authorize tools or declare success without evidence.

The coding path is a bounded ReAct-style executor inside this larger
Planner-Executor/Supervisor architecture. Its actions are parsed into a typed
tool contract, policy checked, executed in the workspace sandbox, and verified.

### 3. Layered memory

Memory is not one undifferentiated vector store:

- short-term: current task context and bounded observations;
- episodic: events, tool results, failures, retries, and replay evidence;
- semantic: durable facts and retrieved knowledge;
- relational: structured links reserved for graph-backed reasoning.

Each durable memory record has a scope, kind, importance, provenance, and
retention policy. Deletion is explicit. Retrieval remains offline and
deterministic by default, with optional embeddings.

### 4. Evidence-based evaluation

Evaluation includes the final answer and the trajectory. AEGIS records:

- correctness and completion;
- latency and token usage;
- tool precision, recall, and efficiency;
- retries and recovery;
- policy violations, tool failures, and hallucination indicators.

This makes it possible to compare a workflow with an agent on the same golden
task rather than assuming that more autonomy is better.

### 5. Governance and operations

The execution boundary keeps typed plan validation, RBAC, policy-as-code,
human approval, sandbox isolation, hash-chained audit, checkpoint/resume,
streaming events, and deterministic verification. MCP/A2A and distributed
backends remain protocol seams; they do not bypass the local policy gateway.

Prompts are versioned as code (`PROMPT_VERSION`), and model/configuration
changes belong in version control with replayable tests.

### 6. Goal and resource contracts

Typed plans may carry a human-owned `GoalSpec`: objective, success criteria,
constraints, risk, human authority, and hard budgets for steps, tools, retries,
wall time, and tokens. The sovereign executor enforces these limits before and
during execution. This turns “autonomy” into bounded goal pursuit rather than
an open-ended loop.

## Deliberate non-goals

AEGIS does not claim that every deployment needs Kubernetes, Kafka, Temporal,
GraphRAG, fine-tuning, or a multi-agent mesh. Those are scaling options. The
local runtime first proves the contract with SQLite, local Ollama, bounded
background jobs, and a deterministic event/audit trail; adapters can replace
those components without changing the contracts.

## Rebuild acceptance criteria

A change is complete only when:

1. the architecture decision is inspectable;
2. tools remain allowlisted and workspace-confined;
3. risky actions pause for approval;
4. every mutation or command has read-back/command evidence;
5. retries are bounded and resumable;
6. traces contain enough data for replay without exposing private reasoning;
7. focused and full offline tests pass.

Model pretraining, reinforcement learning, LoRA, quantization, and speculative
decoding from the supplied systems references are documented as future model
optimization tracks. They are not silently enabled in AEGIS: doing so would
require a training corpus, hardware profile, reproducibility controls, and a
separate safety/evaluation gate.
