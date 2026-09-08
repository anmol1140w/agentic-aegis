# AEGIS document intelligence

AEGIS exposes bounded local document tools through the existing LangGraph tool
registry. `DocumentTools` supports TXT, Markdown, PDF, DOCX, and PPTX discovery,
metadata inspection, section extraction, lexical search, and section reads.

The flow is:

```text
Master → LangGraph tool call → ToolRegistry → workspace/path policy
       → DocumentTools → structured excerpts → graph state → Master review
```

All paths are resolved beneath the configured workspace. Unsupported formats,
missing files, symlink/workspace escapes, oversized inputs, and malformed
queries return structured failures. Results contain a stable document ID,
relative source path, section/page labels where available, bounded excerpts,
offsets for text formats, and truncation metadata.

Document search remains deterministic lexical matching in `DocumentTools`.
For indexed knowledge across sources, `aegis.retrieval.HybridKnowledgeStore`
adds offline hybrid retrieval and source metadata/citations. It does not
require Ollama, a vector database, or network access. Semantic similarity is
optional and requires a caller-supplied embedding callback.

DOCX extraction preserves paragraphs, style/heading names, and table rows.
PDF extraction uses local PyMuPDF; PPTX extraction uses local `python-pptx`.
Document creation/editing remains in the existing artifact pipeline and keeps
its approval and verification behavior.

Example tool calls:

```json
{"action":"tool","tool":"list_documents","arguments":{"path":"."}}
{"action":"tool","tool":"search_documents","arguments":{"query":"P-101 vibration","max_results":5}}
{"action":"tool","tool":"read_document_section","arguments":{"path":"report.pdf","page":2}}
```

The model receives only the returned excerpts and source metadata, not an
unbounded full-document payload. Audit traces record tool names, status,
duration, result size, and source references without storing whole documents.
