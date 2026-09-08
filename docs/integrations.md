# AEGIS agent-workspace integrations

AEGIS now incorporates the portable architectural ideas from the local
OpenCode, Void, and Eigent repositories while keeping one Python execution and
security boundary.

## Reviewable workspace changes

`WorkspaceReadTools` exposes `create_checkpoint`, `list_checkpoints`,
`workspace_diff`, and `restore_checkpoint`. Checkpoints are local tar archives
under `.aegis/checkpoints`, exclude `.git` and the checkpoint store, and are
never restored without explicit approval. Text changes include bounded unified
diffs for review.

## Permission-scoped tool calls

`runtime.permissions.PermissionManager` provides session-scoped allow, prompt,
and deny rules. Deny rules take precedence over prompts and allows. The local
MCP-shaped adapter can use this manager and returns a request ID when a tool
needs approval instead of executing it.

## Skills and workforce execution

`runtime.skills.SkillCatalog` discovers bounded local `SKILL.md` files and
provides metadata or prompt context to a specialist. `runtime.workforce` runs
registered AEGIS specialists concurrently with a bounded parallelism limit;
unknown agents return structured failures rather than being created implicitly.

## Native Bubblewrap sandbox

`tools.native_sandbox.NativeSandbox` provides a persistent, DeepAgents-style
workspace lifecycle on top of AEGIS Bubblewrap. `SandboxConfig` keeps the
network disabled by default, clears inherited environment variables, applies
resource limits when cgroup v2 is available, supports explicit read-only bind
mounts, and exposes a dry-run command representation. File transfer APIs accept
only relative paths that remain under the sandbox workspace.

## Platform integration

External services should integrate through the FastAPI boundary or the typed
contracts in `aegis/`, not by bypassing workspace tools. Use `ExecutionPlan`
for workflow descriptions, `PolicyEngine` for authorization,
`HybridKnowledgeStore` for local cited retrieval, `MemoryStore` for scoped
state, and `AuditChain` for tamper-evident event records. The personal SIH
presentation template does not alter the runtime API.
