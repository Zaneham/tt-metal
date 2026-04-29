# Iteration Subagent

Runs one hypothesis per iteration, per workspace. Dispatched by
`tt:optimizer/SKILL.md` after Baseline, in one of two modes.

## Modes

| Mode | Edits | Rebuild per trial | Hypothesis source |
|---|---|---|---|
| `parameter-search` | Test / config / call-site args | No (Python only) | Enumerated config space |
| `dataflow-optimize` | Op source, kernel code | Yes | Open-ended, grounded in profile |

For `parameter-search`, use when the op has an enumerable config space
(block sizes, shard specs, CB depths, fidelity). For `dataflow-optimize`,
use for open-ended code changes (barriers, CB sizing, NOC batching,
sharding, fusion).

## Inputs

- Target: unit-test path (+ optional `-k` filter) and op identifier.
- Goal: per `convergence.md`.
- Baseline metric + commit SHA + bottleneck tag from the baseline profile.
- Workspace root — this instance operates only inside this path.
- **parameter-search**: enumerated config space (developer-supplied or
  derived from `tt:learn`).
- **dataflow-optimize** parallel only: a hypothesis family label (`-a`,
  `-b`, `-c`) to diversify parallel workspaces.

## Procedure

### 1. Orient

- Read the baseline profile note, the Prepare note, and `tt:learn` notes.
- Read relevant topic files in `tt-agent/knowledge/` (matmul.md, ccl.md,
  sharding.md, ...) — the guidelines there cost prior sessions full
  iterations to discover.
- Identify the op-level bound class (overhead / bandwidth / compute /
  near-peak) from `skills/profiler/interpretation.md`. **Match levers to
  the class** — compute-bound levers on an overhead-bound op waste
  iterations.
- Note which processor is the bottleneck (per-RISC tag), current CB
  depths, current sharding.

### 2. Hypothesize

One concrete change, grounded in profile evidence. Write the hypothesis
in one line:

> "BRISC is 2.1× TRISC1 on matmul_tile — batching 4 reads before the
> single barrier should overlap NOC with compute."

**Bundle when justified, attribute always.** Single-knob change is the
default. Bundle only when parts wouldn't win in isolation (e.g. one
change unlocks the L1 budget another change needs). The hypothesis line
names every knob changed and predicts which contributes most. Bundle
small enough that a regression bisects in one follow-up iter (revert
one knob, re-profile). Subject lists all knobs:
`opt(<scope>): <knob1> + <knob2> + <knob3> — <metric>`.

### 3. Implement

Edit the source (dataflow) or test/config (parameter-search). Keep the
edit tight — only what the hypothesis requires. Do not refactor or fix
unrelated issues.

**Comment hygiene.** A comment passes if it reads sensibly as the first
commit on the branch. Keep correctness arguments and hardware
constraints; drop changelog phrasing (`vs`, `was`, `previously`,
parenthesized obsolete numbers). The commit subject captures the Δ.

On matmul kernel-level assertions, apply the `print(pc)` recipe per
`knowledge/matmul.md` § "Print the realized program config on crashes".

### 4. Build

Invoke `tt:run` to rebuild. For when rebuild is skipped (kernel-only
edits), see `knowledge/recipes/tt-metal/build.md` § Kernel iteration.
parameter-search rarely needs a rebuild after the first trial.

### 5. Run + profile

Invoke `tt:run` to execute the unit test (PCC check runs inside). Invoke
`tt:profiler` for device timing.

### 6. Record

Both artifacts required, in order — skipping `/tt:note` is a discipline failure.

1. **Source-repo commit** with subject `opt(<scope>): <one-line hypothesis> — <metric> (<Δ%> vs best)`.
2. **Note entry** via `/tt:note` topic=`<scope>`, title=`Iter <N> — <one-line hypothesis>`,
   body per `convergence.md` § Entry shape. For parallel workspaces, include
   the workspace letter in the commit cell of the Iterations row.

Then emit one line to the developer in real time:

```
Iter <n> [<ws>] <sha>: <metric> (baseline <B>, Δbest <X%>, best@iter <m>) · <FLOPs%>F / <DRAM%>D / <Bound>
```

### 7. Evaluate

Apply `convergence.md`:
- PCC < 0.999 → abort this workspace only. Others continue.
- Goal met → declare success; main may stop sibling workspaces.
- Stall → main handles the developer prompt; subagents do not.

### 8. Next iteration

**Gate.** Read `~/.tt-agent/notes/<scope>.md`; confirm the HEAD entry matches
the iteration just completed (SHA in body). If absent, invoke `/tt:note` before
continuing — no exceptions. Source commits without matching notes break the
audit contract.

Next hypothesis can be informed by all prior trials (self + siblings).
Read the overview file at each iteration — sibling evidence is signal.

**Pause triggers (this workspace, sliding 5-iter window):**
- 3 reverts → **change approach.** Mode-switch or scope-narrow prompt.
- 3 regressions (no revert) → **re-validate approach.** Pause and check
  direction; no auto-prompt.

Record either in the findings note.

## Forensic commits

Keep failed trials as commits. Label via subject prefix:
- `opt(<scope>): forensic — <…>` (kept-but-failed)
- `opt(<scope>): revert — <…>` (explicit revert)

Filter the audit trail with `git log --grep`.

## Parallel workspace interaction

- **Device access** serialized by tt-device-mcp. No extra coordination.
- **ccache** shared. Parallel builds warm each other's caches.
- **Trend file** shared — each workspace's rows tagged with its letter.
  Writers append; conflicts resolved by re-read-then-append.
- **No cross-workspace code copying** mid-flight. If a sibling finds a
  winning pattern, `tt:code-review` surfaces it post-session.

## Output contract

On goal-met or main-requested stop:
- Winning commit SHA.
- Hypotheses tried, which were productive.
- Untried directions (for manual continuation).

On PCC abort:
- Failing commit SHA + diff vs previous-best.
- Commit kept; do not revert.
