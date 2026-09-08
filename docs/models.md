# AEGIS models and routing — current version

Ollama is the supported inference runtime. Model names are aliases in
`config/models.yaml`; the runtime never hardcodes a model tag in planning.

| Alias | Configured tag | Capability |
|---|---|---|
| `qwen-general` | see `config/models.yaml` | master, general, documents |
| `qwen-coder` | see `config/models.yaml` | coding and repair |
| `qwen-vision` | see `config/models.yaml` | image analysis |
| `llama-small` | see `config/models.yaml` | lightweight capability profile |

The interactive `general_agent` route uses `llama-small` by default so short
terminal requests start quickly. Document, vision, and coding specialists keep
their dedicated configured models. Set `AEGIS_MODEL_TIMEOUT_SECONDS` (default
`90`) and `AEGIS_GENERAL_MAX_TOKENS` (default `256`) in `.env` to tune bounded
native Ollama streaming.

Availability is checked locally at startup. If a configured model is absent,
AEGIS reports a structured unavailable-model result; it does not use a cloud
fallback. The active CLI/API route is Master-first. The lightweight profile is
not an initial handoff and cannot finalize a task without master verification.

## CPU-first profile

`runtime/model_profiles.py` defines bounded context, retry, tool, and latency
limits for local models. The profile is model-agnostic; model size and tags are
owned by `config/models.yaml`.

## Changing models

Edit only `config/models.yaml`, then restart AEGIS:

The registry currently describes chat/vision-capable providers. `code_execution`
is a supported coding capability; `calculation` remains a task capability for
the deterministic calculator and is not advertised by the coder. Embedding
models are intentionally not entries in this registry yet: the current
provider contract is chat-oriented. The local retrieval implementation accepts
an optional embedding callback, but the registry does not yet manage embedding
lifecycle, batching, or health. Add an explicit embedding provider contract
before registering `nomic-embed-text`.

```yaml
qwen-coder:
  model: "your-local-coder-tag"
```

Verify with `/models` or `ollama list`. Do not place credentials or remote API
URLs in the model registry.
