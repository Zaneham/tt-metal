---
name: optimizer
description: "Autonomous kernel/op optimization loop — profiles, hypothesizes, implements, measures, commits per iteration. Operates single or multi-workspace. Use to push a specific op's device time toward a goal."
metadata:
  layer: workflow
---

# TT Optimizer

## Purpose

Drive a performance goal on a specific op or kernel to convergence without
supervision. The loop: profile → hypothesize → implement → profile →
record → repeat. Every iteration lands as a git commit on a dedicated
branch so the trajectory is inspectable and any point recoverable.

Two iteration modes (in `iterate.md`):
- **parameter-search** — discrete sweep over a config space. Single
  workspace, one build, many runs.
- **dataflow-optimize** — open-ended code changes (barriers, CB sizing,
  NOC batching, fusion). One workspace per hypothesis; parallel across
  workspaces is supported.

## When to Invoke

- "Optimize <op> for device time"
- "Get <kernel> to within X% of roofline"
- "Make this matmul faster"
- `tt:orchestrator` dispatches here for "Profile bottleneck, optimize throughput"

## Preflight: developer-rule conflict

This skill commits per iteration and may create new workspaces. Before
starting:

1. State plainly: *"I will commit every iteration to a dedicated branch
   and may create N parallel workspaces. I will never push to a remote."*
2. Check the developer's CLAUDE.md (global and project) for rules
   conflicting with autonomous commits or workspace creation. Surface
   the conflict, quote the rule, ask for override or scope adjustment.
3. Wait for explicit confirmation before proceeding.

## Preflight: target device

Pin the target device (N150, N300, T3K, …) explicitly before baseline.
Ask the developer if not specified. Do not let device-MCP queue routing
silently pick — comparison numbers across devices are not normalised in
the trend table.

## Preflight: tool preload

Fetch these schemas via `ToolSearch` before baseline — loading mid-session
during a CCL hang blocks for the full test timeout (10+ min) before the
reset tool is callable:

- `select:tt_device_reset` — for CCL hangs or watchdog locks.
- `select:tt_device_job_run,tt_device_job_run_bg,tt_device_job_wait,tt_device_job_kill,tt_device_job_logs` — tt:run dispatches through these.

Mandatory when a CCL hypothesis is on the queue. Record load in the first
entry of `<scope>.md`.

## Inputs

- **Target**: op or kernel name + test (pytest path + optional `-k`). If
  the user names only a model, the Extract phase isolates the bottleneck.
- **Goal**: absolute ns, relative %, roofline %, or utilization %. Formulas
  and guidance on picking the right type live in `convergence.md` §
  Success criterion.
- **Mode** (optional): `parameter-search` or `dataflow-optimize`. Default:
  parameter-search if op has an explicit config struct and bounded
  parameter space; else dataflow-optimize.
- **Parallelism** (optional, default 1): concurrent hypotheses in separate
  workspaces. >1 only valid for `dataflow-optimize`.

## Phase Table

| Phase | What happens | Procedure | Note entry |
|---|---|---|---|
| Preflight | Check dev-rule conflicts, preload tools, sweep docs | inline above | — |
| Prepare | Research target, sibling-diff scan, dim validation, doc sweep | `prepare.md` | `Prepare — …` entry to `<scope>.md` |
| Extract* | Capture model inputs, write unit test, verify ±10% | `extract.md` | test path + tensor path |
| Baseline | Profile the unit test | `tt:profiler` | `Baseline iter 0 — …` entry to `<scope>.md` |
| Spawn* | Clone + branch + build per hypothesis | `workspaces.md` | N workspaces, N branches |
| Iterate | Hypothesize → implement → build → profile → commit → record | `iterate.md` (+ `convergence.md`, `knowledge/<topic>.md`) | source-repo commits, `Iter <N> — …` entry to `<scope>.md` per attempt |
| Review | Review-to-done loop on winning branch | `skills/code-review/review-loop.md` | `Reviewed: …` entry to `code-review.md` (caller adds `Reviewed (link) — …` pointer entry to `<scope>.md`) |
| Findings | Final summary of the optimizer session | inline | `Findings — …` entry to `<scope>.md` |

\* conditional — Extract runs only if no unit test exists; Spawn only for parallelism > 1.

After each phase, summarize in 3-5 lines as a new entry in `<scope>.md` and move on.
Phase inputs are consumed, not carried forward.

**Gate.** Do not start a phase until the previous phase's note entry has been
written. The "Note entry" column of the phase table is mandatory, not aspirational —
verify the entry exists before advancing. The same gate applies inside Iterate
(see `iterate.md` § 8).

## Outputs

`~/.tt-agent/notes/<scope>.md` — single timeline file for the entire optimizer
session. **Start here.** The latest entry carries live status (current best,
trend table, stall counter); older entries preserve the history of every
attempt, including forensic failures.

Per phase / per iteration: invoke `/tt:note` with topic=`<scope>` and the
entry body shape from `convergence.md` § Entry shape.

Source repo (tt-metal):

- One commit per iteration on branch `<user>/optimizer/<scope>-<YYYY-MM-DD>[-<letter>]` (`<user>` = `$USER`, namespacing all tt-agent branches).
- Source-repo commit subject format: see `iterate.md` § Record.
- Source-repo commits are never pushed.

## Convergence

See `convergence.md`. Summary:

- Continue while best improved ≥ 2% in the last 5 iterations.
- At 10 iterations without improvement, ask the developer.
- PCC < 0.999 → immediate abort. Commit kept; the latest entry's snapshot rolls back the `best` row to the prior passing commit.

No hard iteration cap. Developer can interrupt anytime; the timeline file and
commits are current after every iteration.

## Caller Contract

- **Developer**: reads `<scope>.md` (latest entry) anytime. On stall prompt,
  responds with direction or stop. At success, cherry-picks or merges
  the winning commit; otherwise deletes branches and workspaces per
  the findings note.
- **Orchestrator**: dispatches here with target + goal; reads the
  findings note on completion.
