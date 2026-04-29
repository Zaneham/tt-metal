# Convergence Rules

Governs when to stop, ask, or abort. Applies to both modes of `iterate.md`.

## State tracked per session

- **baseline**: device time from the Baseline phase. Fixed.
- **best**: lowest device time seen. Updates when a trial beats it.
- **history**: list of `(iter, commit, workspace, metric, pcc)`.
- **stall counter**: iterations since `best` last improved by ≥ threshold.

## Thresholds (defaults, overridable at session start)

| Name | Default | Meaning |
|---|---|---|
| `improvement_threshold` | 2% | Trial is "meaningful" if `(best_prev − trial) / best_prev ≥ 2%`. |
| `window` | 5 | Rolling window. If any of the last N trials beat best by ≥ threshold, keep going. |
| `stall_ask` | 10 | Iterations with no ≥threshold improvement → ask developer. |
| `pcc_abort` | 0.999 | Trial PCC below this → immediate abort. |

## Decision after each trial

- `pcc < pcc_abort` → **abort**. Keep commit for forensics. Findings
  note with failing SHA and last-good-best SHA. Stop.
- trial beats `best` by ≥ `improvement_threshold` → update `best`, reset
  stall counter. **Continue.**
- else → increment stall counter. If any trial in the last `window` beat
  best by ≥ threshold → **continue**. Else if `stall counter < stall_ask`
  → **continue** (no progress this window). Else → **ask developer**.

**Parameter-search early prompt.** 3 consecutive trials in
`parameter-search` mode showing no improvement → prompt the developer
with "switch to `dataflow-optimize`" as the **default** option,
regardless of `stall_ask` counter. `stall_ask=10` remains the final
fallback.

## Success criterion

`best` crosses the supplied goal:

- **Absolute**: `best ≤ target_ns`
- **Relative**: `best ≤ baseline × (1 − target_pct)`
- **Roofline**: `best / roofline_ns ≤ 1 / target_pct` (caller supplies
  `roofline_ns`; see `tt:learn`).
- **Utilization**: `flops_pct ≥ target_flops_pct` AND `target_op_fw_ns ≥
  sum(non_target_op_fw_ns)` (target op is THE bottleneck). `flops_pct` /
  `dram_pct` come from `tt-perf-report` (see
  `skills/profiler/interpretation.md`). Use when baseline is low-utilization
  — a 30% speedup at 25% FLOPs hasn't fixed the op.

On success: prepend a `Findings — …` entry to `<scope>.md`, invoke
`tt:code-review` via `review-loop.md` on the winning branch, report.

## Bound-ceiling exit (utilization-goal only)

If the op's bound class stays `overhead / sync-bound` with FLOPs% < 40%
after 3 productive iterations (FLOPs% under 35% in every trial, and
remaining levers — `in0_block_w`, `per_core_M`, L1 sharding — tried or
ruled out by L1 budget), write a `kernel-family-ceiling` checkpoint and
ask:

```
Reached kernel-family ceiling at iter <N>. Best <best> at <flops%>F /
<dram%>D / overhead. Current variant (<2D MC | 1D ring | DRAM-sharded>)
appears structurally overhead-bound on this shape.

Options:
  1. Switch to <alternative variant> as a parameter-search sweep.
  2. Accept current best (structural change out of scope).
  3. Broaden scope to adjacent ops (AllGather, tilize, ...).
```

Prevents long subblock fiddling on a misclassified goal.

## Constraint feasibility probe

When a new constraint enters scope mid-session (e.g. PCC bar raised, new
correctness gate, accuracy floor added), measure it at the baseline
commit before adopting as a goal. If baseline already fails the
constraint, declare it upstream/downstream-bound and out of scope — do
not pursue from the current op. Record the probe result in `<scope>.md`.

## Stall prompt

```
Stalled at iteration <N>. Best: <best> (baseline <baseline>, Δ <pct>%).
Last <stall_ask> iterations improved best by less than <threshold>%.

Recent trials:
<table of last 5: commit, hypothesis, metric>

Timeline: ~/.tt-agent/notes/<scope>.md (latest entry)
Workspaces: <list>

Options:
  1. Continue for another <stall_ask> iterations.
  2. Switch mode (parameter-search ↔ dataflow-optimize).
  3. Narrow scope.
  4. Stop and accept current best.
```

Wait for developer choice. On timeout, keep state and stay idle.

## PCC abort forensics

1. Keep the failing commit — do not revert.
2. Findings note: failing SHA, diff vs previous-best, exact PCC value,
   note to treat as evidence (not a fix).
3. The latest entry's snapshot keeps the `best` row at the last passing commit.
   Workspace HEAD is at failing commit for easy inspection.

## Entry shape — status first, then change

Every entry in `<scope>.md` follows this body shape: `### Status` first
(carrying the live-state snapshot at this point in the trajectory),
`### Change this step` below. There is no separate overview file — opening
`<scope>.md` and reading the latest entry top-to-bottom gives at-a-glance
state immediately, then what was done.

```markdown
## <iter or phase title>
**<YYYY-MM-DD HH:MM>** · `<source-repo>@<short-sha>`

### Status

| Best | Baseline | Δ baseline | Iter | Stall | Util (best) | Bound | Goal |
|---|---|---|---|---|---|---|---|
| <ns> @ iter <m> ws <letter> `<sha>` | <ns> | −<pct>% | <n> | <s>/<stall_ask> | <flops%>F/<dram%>D | <bound> | <goal kind> |

#### Iterations (chronological — most recent at top)

| # | Time | Commit (ws) | Metric | PCC | Util | Δbest | Hypothesis |
|---|---|---|---|---|---|---|---|
| <N> | HH:MM | `<sha>` (<ws>) | … | … | … | … | … |
| <N-1> | … | … | … | … | … | … | … |
| 0 (baseline) | … | … | … | … | … | — | as-is |

#### Per-iteration contribution

| # | Change | Saved | % of baseline | Running total |
|---|---|---|---|---|

#### Parameter sweeps (per knob, when non-monotonic)

| Knob | Values tried | Best value | Notes |
|---|---|---|---|

#### Op-level timing (context, optional)

| Op | Baseline | Current | Δ |
|---|---|---|---|

### Change this step

**Hypothesis:** <one-line>
**Result:** Metric <ns> (Δbest <pct>) · PCC <p> · <flops%>F/<dram%>D/<bound>
```

Rule: anything shown to the developer in chat must also land in the Status
section's tables. Each new entry regenerates Status with the trajectory up to
that point; older entries preserve their snapshots verbatim, which is
informative — the historical trajectory is recoverable by scrolling. Truncate
the Iterations table to the last 50 rows once it exceeds 100 (preserve
baseline and current best); older rows remain accessible via
`git log -- <scope>.md` and `git show`.

Forensic-failure rows are inline in the Iterations table
(`| <N> (forensic) | … | CRASH | — | — | — | <reason> |`). Filter via
`git log --grep "(forensic)" -- <scope>.md`.

## Interruption

Each iteration writes its entry and commits synchronously. Resume by
re-reading the latest entry of `<scope>.md` and the branch HEAD — invoking
the optimizer with the same scope and goal picks up from current best.
