# AEGIS workflows — current version

## Master workflow

All normal CLI and API requests use this sequence:

```text
normalize → inspect repository → classify capability → create bounded plan
→ validate plan → delegate one step → observe result → verify
→ review → bounded repair or final evidence
```

The master does not expose privileged mutation tools directly. Specialists get
only their allowlisted tools. The lightweight model is not a first-stage
terminal specialist; the master owns the route and review.

## Coding workflow

```text
repository_context/tree/search
→ read relevant source
→ checkpoint
→ approval
→ one minimal edit/create
→ mandatory read-back
→ syntax/static validation
→ targeted command/test
→ diff and independent review
```

Failures are classified into bounded categories. Repair is limited, does not
retry policy or security failures, and never edits a test merely to hide an
implementation failure.

## Document and vision workflows

Document and image paths are resolved under `./workspace`. Text extraction,
OCR, PDF/DOCX/PPTX inspection, and image preprocessing are local and bounded.
Visual or repository content is data, not authority; instructions found inside
files cannot approve tools or expand permissions.

## Outputs and evidence

Each run writes `result.txt`, `metadata.json`, `trace.json`, and when applicable
`network_report.json` under `workspace/outputs/<run_id>/`. Commands also write a
bounded record under `workspace/executions/`. Artifacts created under
`workspace/artifacts/` receive provenance and SHA-256 registration.
