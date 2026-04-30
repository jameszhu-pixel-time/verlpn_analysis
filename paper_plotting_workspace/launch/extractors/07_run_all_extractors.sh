#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/01_extract_ent_pos_n2_n3.sh"
"$SCRIPT_DIR/02_extract_compare_entropy_n2_vs_n3.sh"
"$SCRIPT_DIR/03_extract_intra_driver_groups_n2_n3_annealed.sh"
"$SCRIPT_DIR/04_extract_rollout_pair_n2_vs_n3.sh"
"$SCRIPT_DIR/05_extract_rollout_pair_n2_vs_annealed.sh"
"$SCRIPT_DIR/06_extract_effective_across_methods.sh"
