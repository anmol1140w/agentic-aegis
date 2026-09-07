# AEGIS models and routing — current version

Ollama is the supported inference runtime. Model names are aliases in
`config/models.yaml`; the runtime never hardcodes a model tag in planning.

| Alias | Default tag | Capability |
|---|---|---|
| `qwen-general` | `qwen3.5:9b` | master, general, documents |
| `qwen-coder` | `qwen2.5-coder:7b` | coding and repair |
| `qwen-vision` | `qwen3-vl:8b` | image analysis |
| `llama-small` | `llama3.2:1b` | lightweight capability profile |

Availability is checked locally at startup. If a configured model is absent,
AEGIS reports a structured unavailable-model result; it does not use a cloud
fallback. The active CLI/API route is Master-first. The lightweight profile is
not an initial handoff and cannot finalize a task without master verification.

## CPU-first profile

`runtime/model_profiles.py` defines bounded context, retry, tool, and latency
limits for local 4–7B-class models. `qwen2.5-coder:7b` remains the coding
compatibility default; the profile is model-agnostic.

## Changing models

Edit only `config/models.yaml`, then restart AEGIS:

```yaml
qwen-coder:
  model: "your-local-coder-tag"
```

Verify with `/models` or `ollama list`. Do not place credentials or remote API
URLs in the model registry.
