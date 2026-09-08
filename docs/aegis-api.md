# AEGIS API — Current Status

AEGIS — Sovereign Agent Workbench exposes a thin FastAPI boundary around the
existing `Orchestrator.run_master` implementation. The API does not contain
routing, specialist prompts, policy, or tool execution logic; those remain in
the runtime used by the CLI.

Run locally with:

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

From the repository root, a typical local setup is:

```bash
source .venv/bin/activate
python cli.py                 # interactive terminal client
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Example API calls:

```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/tasks \
  -H 'content-type: application/json' \
  -d '{"request":"list files in the workspace","files":[],"options":{}}'
curl http://127.0.0.1:8000/api/tasks/<execution_id>
curl -N http://127.0.0.1:8000/api/tasks/<execution_id>/events
```

Endpoints:

- `GET /api/health` — reports service and actual configured model availability.
- `POST /api/tasks` — queues `{ "request": "...", "files": [], "options": {} }`.
- `GET /api/tasks/{execution_id}` — returns current/final status and result.
- `GET /api/tasks/{execution_id}/events` — streams high-level SSE progress.

The service is local-only by default. Django should call this HTTP boundary;
it should not import AEGIS internals. File metadata is passed as context and
is still subject to the runtime workspace and policy boundaries.

## Platform endpoints

Validate and register typed workflows through `/api/workflows/validate` and
`/api/workflows`. Knowledge and memory use `/api/knowledge`,
`/api/knowledge/search`, `/api/memory`, and `/api/memory/search`.
Operational endpoints include `POST /api/jobs`, `GET /api/jobs/{job_id}`,
`DELETE /api/jobs/{job_id}`, `GET /api/audit/verify`, and
`GET /api/evaluations/summary`.

The current API principal adapter is local-admin for the prototype. Replace it
or front it with enterprise identity before deployment to multiple users.

## Terminal client

The same Master runtime is available without HTTP:

```bash
source .venv/bin/activate
python cli.py
```

Use the `You ▶` prompt for workspace, coding, document, or vision requests.
Interactive `/models`, `/network`, and `/quit` commands are handled locally;
write and approved-command operations pause for terminal confirmation. The
CLI and API persist result and trace artifacts under
`workspace/outputs/run_<id>/`.
