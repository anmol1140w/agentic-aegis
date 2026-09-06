# AEGIS Tool Development — Current Status

Add a parameterized primitive to `tools/`, register it in `ToolRegistry`, and
provide explicit risk/permission/approval/offline/timeout metadata. Return a
structured result (`status`, `data`, `artifacts`, `evidence`, `metadata`, and
bounded error) where practical. Add unit, policy, failure, composition, and
trace tests. Never expose unrestricted shell or bypass workspace validation.
Keep network, external side effects, Git mutation, and RAG disabled unless a
future policy explicitly enables them.
