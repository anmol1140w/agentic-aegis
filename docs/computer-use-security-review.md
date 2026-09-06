# AEGIS computer-use security review

Status: **disabled by default**.

AEGIS does not currently register unrestricted desktop, mouse, keyboard, or
screen-control tools. This is deliberate: a screenshot is untrusted input and
must never authorize an external or consequential action. Before enabling a
computer-use capability, the implementation must add all of the following to
the existing LangGraph/tool runtime:

- an explicit display/window/application allowlist;
- bounded screenshot regions and history limits;
- structured actions (`click`, `type`, `keypress`, `scroll`, `wait`);
- coordinate/target validation and an emergency stop;
- per-action and total iteration/time budgets;
- confirmation for submissions, messages, deletion, permissions, or external
  side effects;
- cancellation propagation and process cleanup;
- audit events for capture, proposal, approval, execution, and observation;
- visual prompt-injection handling that treats on-screen text as data;
- tests proving capture → inference → validation → action → observation.

Until those controls and tests exist, visual analysis remains read-only and
local. No browser or desktop-control dependency is required for AEGIS.
