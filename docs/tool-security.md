# AEGIS Tool Security — Current Status

Policy is deterministic infrastructure: workspace paths are resolved and
checked before I/O; read tools are automatic; edits and approved commands use
the existing approval and command allowlists. Delete, arbitrary shell,
network, credentials, Git mutation, and workspace escape are denied or remain
disabled. Tool metadata records risk and permissions for discovery, but cannot
grant authority beyond the policy layer. All execution stays local/Ollama-only
by default.
