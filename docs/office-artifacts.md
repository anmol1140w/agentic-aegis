# AEGIS Office Artifacts

AEGIS generates PowerPoint and Excel files locally through the Master Agent. The
model selects a capability and produces intent only. Deterministic Python tools
convert a bounded structured specification into a `.pptx` or `.xlsx`, reopen the
file, and validate the package before the Master reports success.

## Architecture

```text
request → task graph → capability route → specialist → structured spec
         → policy gateway → deterministic artifact service → validation
         → bounded repair / audit → workspace output
```

The production integration uses:

- `runtime.agents.MasterAgent` and `runtime.task_graph` for routing, delegation,
  verification, and bounded repair.
- `tools.artifacts.ArtifactService` as the policy-facing subsystem.
- `tools.artifacts.presentation` with `python-pptx` for PPTX files.
- `tools.artifacts.spreadsheet` with `openpyxl` for XLSX files.
- `runtime.tool_policy.PolicyEngine` for workspace confinement and approvals.
- `security.audit.AuditLogger` for operation metadata without model reasoning.

## Structured specifications

Presentation specifications contain `title`, optional `subtitle`, `theme`, and a
non-empty `slides` list. Slides can contain `layout`, `title`, `bullets`,
`columns`, `table`, and `chart`. Supported themes include `simple`,
`professional`, `technical`, `business`, `academic`, `dark`, and `light`.

Workbook specifications contain `title` and `worksheets`. Each worksheet can
contain `name`, `headers`, `rows`, `cells`, `formulas`, `table`, `charts`,
`freeze_panes`, `merged_cells`, and `number_formats`.

The model does not emit Python, manipulate ZIP packages, or choose an unrestricted
filesystem path.

## Validation

PPTX validation checks the file, ZIP package, `python-pptx` parsing, slide count,
titles, and empty-slide warnings. XLSX validation checks the file, ZIP package,
`openpyxl` parsing, expected worksheet names and cells, tables, and charts.
Failed validation produces a structured failure and cannot be reported as a
successful artifact.

## Security and audit

All source, template, image, and output paths must resolve inside the configured
workspace. Traversal, absolute-path escapes, and symlink escapes are rejected.
Artifact creation and editing are approval-gated through the existing policy
engine. Normal generation does not use network access. Audit records include
operation, artifact type, path, validation, status, duration, and bounded
metadata. Private chain-of-thought is never persisted.

## CLI usage

Normal requests use the existing REPL:

```text
you > Create a 5-slide presentation explaining AEGIS
you > Create an Excel budget workbook with a chart
you > Analyze this dataset and create both an Excel workbook and a PowerPoint summary
```

Useful deterministic commands:

```text
/validate-artifact workspace/outputs/artifacts/file.pptx
/inspect-artifact workspace/outputs/artifacts/file.xlsx
/artifacts
```

Generated files are stored below the approved workspace, normally under
`workspace/outputs/artifacts/`, and each Master run also persists its result and
trace under `workspace/outputs/<run_id>/`.

## Current limitations

The initial deterministic subsystem supports common text, tables, editable
charts, formulas, freeze panes, and basic editing operations. It does not yet
preserve arbitrary PowerPoint theme masters or every native Excel feature when
editing an existing file. Unsupported operations return structured errors rather
than silently changing the source.
