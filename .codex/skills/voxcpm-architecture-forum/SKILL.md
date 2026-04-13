---
name: voxcpm-architecture-forum
description: Moderation rules for VoxCPM.cpp architecture review sessions and long-horizon Torch-to-GGML refactor decisions. Use when discussing migration direction, deciding whether to keep/delete bridge code, spawning subagents for a design review, or preventing the repository from drifting into local-optimum complexity.
---

# VoxCPM Architecture Forum

Use this skill when the task is primarily discussion, planning, architecture review, or long-horizon refactor control rather than immediate feature coding.

The purpose is not to generate more implementation ideas. The purpose is to keep VoxCPM.cpp converging toward a simpler runtime shape that resembles a mature GGML project.

## Quick Start

1. Read `AGENTS.md`.
2. Read `VOXCPM_RUNTIME_REFACTOR_PLAN_zh.md`.
3. Read only the minimum relevant migration references.
4. If the discussion touches runtime, backend, state, output, graph cache, or decode hot paths, run:

```bash
./.codex/skills/voxcpm-runtime-migration-guard/scripts/audit-runtime-boundaries.sh
```

5. Classify the meeting before doing anything else:
   - diagnosis
   - deletion review
   - stage-gate review
   - architecture decision
   - implementation stop-check

## Meeting Goal

The default goal is **complexity reduction with preserved correctness**, not “maximum local numerical progress”.

A good meeting outcome does one or more of these:

- removes or freezes a bad boundary
- narrows a public API
- turns an implicit bridge into an explicit temporary fallback
- raises a stage gate that blocks premature optimization
- identifies code that should move out of runtime and into tests/examples/tools

## Non-Negotiables

- Trace alignment is evidence, not architecture.
- A temporary bridge is not allowed to quietly become the default API.
- Prefer deleting, hiding, or freezing an interface over extending it.
- Do not reward “it works” if it expands host/device boundaries or public surface area.
- Do not add new public runtime helpers whose only job is serving one benchmark or one intermediate debug path.
- If a proposal adds prose to the plan, it must change boundary, ordering, ownership, or acceptance criteria. Otherwise update only progress.

## Discussion Roles

If subagents are used, keep them lens-specific and independent. Good default lenses are:

- architecture convergence
- hot-path transfer and memory traffic
- llama.cpp / whisper.cpp best-practice comparison
- validation methodology
- API surface and deletion candidates
- execution method and anti-drift rules

Use at most 6 subagents.

## Proposal Filter

For every proposed change, answer these questions before endorsing it:

1. Which permanent runtime boundary does this clarify?
2. Does it reduce or increase host-visible intermediate data?
3. Should this live in public runtime API, internal runtime API, test utility, or example code?
4. If it is temporary, what exact future change deletes it?
5. What metric would show this made the repository simpler?

If these questions cannot be answered crisply, the proposal is probably premature.

## Default Biases

- Prefer backend-resident state handles over host vectors.
- Prefer one stable runtime path over many helper wrappers.
- Prefer small internal adapters over broad public compatibility layers.
- Prefer contract tests, state-lifetime tests, and end-to-end checks over multiplying benchmark-style helper entry points.
- Prefer rewriting a confused boundary over preserving it for sentimental reasons.

## Stop Conditions

Pause implementation and return to planning if any of these happen:

- the easiest next step adds a new `tensor_get -> std::vector<float> -> tensor_set` bridge
- a benchmark helper starts dictating runtime API shape
- the plan changelog grows faster than the runtime becomes simpler
- two “temporary” paths now need to be kept in sync
- a discussion about optimization begins before state/output ownership is settled

## Expected Output

A good architecture-forum response should end with:

1. a short diagnosis of why the repository is drifting
2. a list of boundaries to keep, freeze, hide, or delete
3. a staged execution rule set
4. a small number of measurable success signals

Do not end with a long implementation backlog unless the meeting explicitly asks for one.
