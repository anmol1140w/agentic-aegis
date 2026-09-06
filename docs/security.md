# AEGIS Security Architecture & Current Status

Security is enforced deterministically by policy and tools. Workspace paths
are canonicalized before I/O; reads are automatic; edits and allowlisted test
commands require approval. Network, credentials, deletes, Git mutation,
arbitrary shell, and workspace escapes are denied. All model inference and
OCR remain local. These controls apply equally to CLI and FastAPI executions.

---

## 1. Threat Model & Boundaries

Sensitive government, PSU, and defence organizations operate under strict data handling mandates. The workbench assumes an adversarial model where:
1. Generated code or untrusted input could attempt filesystem traversal.
2. Compromised packages or dependencies could attempt external telemetry.
3. User prompts could attempt to invoke high-risk system commands.

```text
┌─────────────────────────────────────────────────────────────┐
│                     SECURITY PERIMETER                      │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Network Isolation: Localhost / Air-gapped only      │   │
│   │ Live OS Socket Audit: external_connections == 0     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Path Validation:                                    │   │
│   │  - Enforces workspace root (./workspace)            │   │
│   │  - Blocks symlink traversal & .. path escapes       │   │
│   │  - Blocks system paths (/etc, /root, /proc, /sys)   │   │
│   │  - Blocks executable file creation (.sh, .exe, .so) │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Sandboxing (Docker):                                │   │
│   │  - network_disabled = True                          │   │
│   │  - read_only root filesystem                        │   │
│   │  - cap_drop = ["ALL"]                               │   │
│   │  - mem_limit = "256m", CPU quota limits             │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Audit & Accounting:                                 │   │
│   │  - Append-only JSONL log (logs/audit.jsonl)         │   │
│   │  - Excludes raw confidential document payloads      │   │
│   │  - Records model, tool, runtime, status, errors     │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Path Sandboxing (`security/permissions.py`)

All file tools (`read_file`, `write_file`, `list_files`, `create_directory`) must validate target paths through `validate_path()`:

* **Boundary Confinement:** Verifies `resolved_path.relative_to(workspace_root)`. If a path resolves outside the workspace, a `PermissionError_` is raised.
* **Blocked System Segments:** Explicit inspection for substrings like `/etc/`, `/root/`, `/proc/`, `/.ssh/`, `/.aws/`.
* **Symlink Resolution:** Follows symlinks and guarantees the real target resides strictly within the workspace.
* **Dangerous Extensions:** Prohibits writing scripts and binaries (`.sh`, `.bash`, `.exe`, `.so`, `.dll`, `.ps1`).

---

## 3. Real Network Auditing (`security/network.py`)

Unlike mock indicators, `NetworkMonitor` queries the Linux kernel TCP/UDP connection table using `psutil.net_connections(kind="inet")`:

1. **Loopback Classification:** Any connection to `127.0.0.1`, `::1`, or `localhost` is marked as local.
2. **Private LAN Classification:** Any connection matching RFC-1918 subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`) is marked as LAN.
3. **External Connection Alarm:** Any connection outside loopback/LAN increments `external_connections`. If non-zero during air-gapped demonstration, security alerts are raised.

---

## 4. Audit Logging (`security/audit.py`)

All lifecycle events write to an append-only JSONL file at `logs/audit.jsonl`:

```json
{
  "timestamp": "2026-09-04T13:14:15.123456Z",
  "event": "model_invoked",
  "task_id": "a1b2c3d4",
  "user": "local",
  "model": "qwen2.5-coder:7b",
  "tool": null,
  "action": null,
  "status": "success",
  "duration_ms": 1420.5,
  "metadata": {"task_type": "coding"},
  "error": null
}
```
