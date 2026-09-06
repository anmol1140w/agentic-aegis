# AEGIS Tool Registry — Current Status

`tools.registry.ToolRegistry` is the single in-process catalog. Each entry
has a callable schema plus category, risk, approval, permissions, timeout,
offline, reversible, and idempotent metadata. `search`, permission/risk
filters, and `check_available` support bounded discovery.

The orchestrator registers calculator, file/document/OCR primitives, and
workspace read/edit/command handlers. Specialist descriptors further narrow
which registered tools they may use. Unknown tools are rejected before
execution.
