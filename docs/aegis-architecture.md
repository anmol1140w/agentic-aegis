# AEGIS execution architecture

AEGIS uses a bounded local execution protocol:

```text
request → repository index → bounded context → plan → validate → execute
        → observe → deterministic verification → independent review → evidence
```

The small local model proposes narrowly scoped actions. Python owns state,
workspace policy, tool contracts, retries, checkpoints, and completion evidence.
The SQLite repository index extracts file metadata, Python/common-language
symbols, imports, and test candidates. `aegis.retrieval.HybridKnowledgeStore`
provides a separate offline knowledge store with lexical ranking and optional
embedding similarity. `runtime/model_profiles.py`
defines the model-agnostic CPU-first 5B operating limits and keeps the existing
the configured coder model; model tags are owned by `config/models.yaml`.

Security and policy failures are non-retryable. Recoverable failures are
normalized into the failure taxonomy and may use at most the configured repair
budget. The deterministic reviewer checks changed-file evidence and verification
before a task can complete.

Offline benchmark scoring is available through `benchmarks/runner.py`; it can
compare `coder_only` and `aegis` runners without contacting a model provider.

The SIH presentation workflow uses the supplied six-slide template and is
documented in `docs/aegis-platform.md` and the repository README.
