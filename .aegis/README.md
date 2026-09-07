# AEGIS workspace control plane

`.aegis` contains machine-facing policy, skills, tool metadata, and runtime
artifacts. The repository README remains the human product overview. Tools are
implemented in Python; this directory contains their discoverable contracts
and procedural guidance.

Use the canonical namespaced runtime (`WorkspaceReadTools.invoke_tool`) for new
integrations. Legacy names such as `read_file` and `edit_file` remain aliases
for existing agents.
