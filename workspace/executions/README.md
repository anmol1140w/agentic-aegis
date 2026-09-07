# Execution state

This directory stores bounded execution state and metadata for the local
Master-Agent runtime. Files are runtime-generated; do not place secrets or
unrelated user data here.

Every successful or failed `workspace.execute_command` call writes one JSON
record containing the command, workspace-relative cwd, exit code, bounded
stdout/stderr, duration, timeout state, and execution ID. The example file
documents the record shape without pretending to be a live run.
