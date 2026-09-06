# AEGIS local vision subsystem

AEGIS exposes local image analysis through the existing LangGraph tool runtime.
`VisionRuntime` validates the input path against the configured workspace,
preprocesses one bounded image, sends it to the configured Ollama vision
provider, and returns structured observations to graph state.

```text
Master → LangGraph → ToolRegistry → path/policy checks
       → VisionRuntime → Ollama qwen3-VL → structured observations
       → graph state → review
```

Current tools:

- `analyze_image(path, task)` — one bounded local image inference;
- `compare_images(paths, task)` — one inference over at most two local images.

Results include task type, source identity, observations, confidence when
provided by the model, uncertainty, preprocessing metadata, model name,
duration, and structured errors. Visual text is treated as untrusted content;
it cannot authorize a tool or computer action. Raw image bytes remain local
and are not written to audit traces.

The existing deterministic preprocessor performs RGB conversion, safe margin
handling, aspect-preserving downscaling, mild contrast/sharpening, and a
configurable size limit. Timeouts, missing models, invalid paths, corrupt media,
and provider failures become structured failures.

Computer-use actions, desktop capture, and arbitrary mouse/keyboard control are
not enabled by default. If introduced later, they must be separate explicitly
approved tools over the same policy and audit boundary, with confirmation for
consequential actions. No cloud vision API, remote MCP server, embeddings, or
RAG is required by the current implementation.

