---
name: aegis-repository-analysis
description: Understand a repository before proposing or making changes.
---

Use `workspace.read.context`, `workspace.read.tree`, `workspace.read.search`,
and `repo.intel` data. Bound the result to relevant files, symbols, imports,
tests, configuration, and Git state. Do not read the entire repository into a
model prompt.
