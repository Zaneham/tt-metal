# Prepare phase

Runs once per session, before the first baseline profile. Produces a short
note in `~/.tt-agent/notes/` that the Baseline phase reads.

## Steps

1. **Research the target op.** Invoke `tt:learn("<target> — config knobs,
   valid dim constraints, kernel variants")`. Record the note path in the
   overview file. The first iteration's hypothesis queue depends on this.

2. **Sibling-implementation scan.** If the target is "reused from X with
   changes" (very common in tt-metal — most multimodal and model variants
   extend a Llama reference), read X and diff against the target. The
   original's config patterns may not apply to the new model's dimensions.
   Note any divergent dims, config lambdas, or cached `args.*` fields.

   **Check the sibling's mode branch** before borrowing structural choices.
   tt-metal models often branch on `mode=="prefill" | "decode"` with very
   different memory layouts. `grep` the sibling for the mode keyword and
   trace which progcfg the target's shape would select. See
   `knowledge/matmul.md` § "Verify a sibling's mode before borrowing
   structural choices".

   **Sibling-divergence guard.** If sibling and target diverge on M, K, N,
   num_chunks, or `fuse_batch` path, treat the sibling as a counter-example
   on those axes — list each divergence in the Prepare note and do **not**
   carry the sibling's structural choices across them.

3. **Actual-vs-derived dim check.** For every `args.<dim>` used in a
   progcfg, compare to `self.<weight>.shape[-1]` of the loaded tensor.
   `ModelArgs` fields are computed at config time and can truncate
   (int-div) or drift from actual weight shape. Prefer the weight-shape
   value in progcfgs. If a mismatch exists, flag it in the overview file —
   a standalone fix PR may be warranted.

4. **Shape-transform proposals.** For each matmul in scope, enumerate the
   divisor structure of M_t, K_t, N_t. Where prime factors of K_t prevent
   friendly `in0_block_w` divisors, propose padding hypotheses on the
   upstream matmul's N (= K of next), or on M / K directly. Alternate
   sharding axes (BLOCK ↔ HEIGHT ↔ WIDTH) are co-equal levers. Padding
   hypotheses surface in the iter-1 queue, not after stall — they reshape
   the parameter space. See `knowledge/matmul.md` § "Padding for divisor
   structure".

5. **Authoritative docs sweep.** Ask the developer whether internal guides
   exist (Confluence pages, `tech_reports/`, PDF exports in the workspace).
   For matmul the PSE Matmul Configuration Guide governs variant selection,
   L1 budgeting, and subblock rules — reading it before iterating reshapes
   the hypothesis queue substantially.

6. **Topic knowledge skim.** Skim `knowledge/matmul.md` (and `ccl.md`,
   `sharding.md` if relevant). Entries there cost prior sessions full
   iterations to discover.

## Output

Invoke `/tt:note` with topic=`<scope>`, title=`Prepare — <one-line summary>`,
body=<body below>. tt-learn (Step 1) has already written its own entry — this
phase produces two commits, one per entry.

Conventional body:

```markdown
**Target:** <op> · test: <test path> -k <filter>

**Research pointer:** see `learn-<subject>.md` (entry written this timestamp).
**TL;DR:** <1–2 sentence summary of the research>

**Sibling-implementation diff:** <summary of divergences; "n/a — not reused" if standalone>
**Dim validation:** <actual-vs-derived mismatches; "all match" if clean>
**Mode-branch confirmation:** <if sibling has PREFILL/DECODE split: which branch target inherits; n/a otherwise>
**Doc pointers:** <internal guides, PDFs in workspace, tech_reports/ references>
```

The Baseline phase reads this entry from `<scope>.md`.
