#!/bin/bash
set -e
cd /data/sanjaysundaram/tt-metal

# Commit 1: test_api.yaml
git add tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_api.yaml
git commit -m "Fix test_api.yaml: remove obsolete eth_coord_mapping and allocation_policies

The physical_mesh.eth_coord_mapping field and allocation_policies section
were removed from the tt_fabric YAML parser but this reference file was
never updated, making it unrunnable and misleading (issue #29448).

Removed the stale sections and clarified that this file is an API
reference document, not an executable test config."

# Commit 2: test_fabric_deadlock_stability_bh_6U_galaxy.yaml
git add tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_deadlock_stability_bh_6U_galaxy.yaml
git commit -m "Improve docs for disabled 2D Torus Y test in BH 6U Galaxy deadlock stability

Added explicit root-cause description (credit starvation in Y-direction
routing) and re-enable criteria to the disabled UnicastAlltoAll Torus Y
test. Tracked in issue #33456."

# Commit 3: run_cpp_fabric_tests.sh
git add tests/scripts/run_cpp_fabric_tests.sh
git commit -m "Improve disabled test documentation in run_cpp_fabric_tests.sh

Replaced vague TODO comment with explicit explanation of why slow dispatch
fabric tests are disabled (no suitable runner pool) and added re-enable
criteria. Tracked in issue #24335."

# Commit 4: run_t3000_unit_tests.sh
git add tests/scripts/t3000/run_t3000_unit_tests.sh
git commit -m "Improve disabled test documentation in run_t3000_unit_tests.sh

Same fix as run_cpp_fabric_tests.sh: replaced vague TODO with explicit
root cause and re-enable criteria for slow dispatch tests.
Tracked in issue #24335."

# Commit 5: t3000-fast-tests-impl.yaml
git add .github/workflows/t3000-fast-tests-impl.yaml
git commit -m "Document disabled multiprocess tests in t3000-fast-tests-impl.yaml

Added explicit root-cause description (race conditions during multi-device
startup) and owner attribution. Added TODO to file a proper tracking issue
since none currently exists."

# Commit 6: galaxy_unit_tests.yaml
git add tests/pipeline_reorg/galaxy_unit_tests.yaml
git commit -m "Improve disabled test docs in galaxy_unit_tests.yaml (issue #44339)

Expanded terse one-line comments into structured disable blocks with:
- Root cause descriptions (coordinate bounds error, kernel compilation hang)
- Full issue URL links
- Explicit re-enable criteria"

# Commit 7: blackhole_e2e_tests.yaml + blackhole-multi-card-unit-tests-impl.yaml
git add tests/pipeline_reorg/blackhole_e2e_tests.yaml .github/workflows/blackhole-multi-card-unit-tests-impl.yaml
git commit -m "Improve disabled BH prefetcher test docs (issue #41040)

Expanded terse disable comment in both blackhole_e2e_tests.yaml and
blackhole-multi-card-unit-tests-impl.yaml with root-cause description
(prefetcher buffer management issue) and re-enable criteria."

echo ""
echo "All 7 commits created successfully on branch: $(git branch --show-current)"
echo ""
git log --oneline -7
