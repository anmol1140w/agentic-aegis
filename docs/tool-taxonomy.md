# AEGIS Tool Taxonomy — Current Status

This inventory follows the common patterns in MCP (discoverable, schema-driven
tools), OpenAI tool definitions (many-to-many agent/tool bindings), LangGraph
tool nodes (bounded state transitions), and NIST trustworthy-agent guidance.
The useful abstraction is a small parameterized primitive with explicit policy
metadata, rather than one agent per file format.

## Families and priorities

| Family | Representative primitives | Priority |
|---|---|---|
| Filesystem | read/list/search/find, metadata, write/edit, safe delete | P0 (read/edit) |
| Documents/PDF | document runner, OCR, text/metadata extraction, validate, DOCX creation | P0 |
| Vision/images | preprocess, OCR image/PDF, inspect image, resize/validate | P0 |
| Text/NLP | normalize text/OCR, entities, keywords, identifiers, intent | P0 |
| Code | source read/search, edit, diff, targeted pytest | P0 |
| Artifacts | create/register/list/find/validate with provenance | P0 |
| Git | status/diff (read-only); branch/commit/push | P1/P2, gated |
| Data | JSON/CSV load/save and deterministic analysis | P1 |
| Archives | create/extract/list/validate with traversal guards | P1 |
| Database | schema/read/query; writes require approval | P1/P2 |
| Spreadsheet/presentation | workbook/cell/chart and slide primitives | P1 |
| Web/API/browser/communication | HTTP, browser, email, calendar, hosting | P3, disabled offline |
| Memory/RAG | session state; retrieval/embeddings/vector search | P2/P3; RAG deferred |
| Security/observability | policy checks, approvals, trace/metrics | P0 |

Tools are discovered by registry search and bound only to the specialist that
needs them. External/network families remain disabled by default.

Sources: [MCP tools specification](https://modelcontextprotocol.io/specification/draft/server/tools),
[OpenAI Agents tools](https://openai.github.io/openai-agents-python/tools/),
[LangGraph tool calling](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/),
and [NIST AI Agent Standards Initiative](https://www.nist.gov/artificial-intelligence/ai-agent-standards-initiative).
