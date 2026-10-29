#!/usr/bin/env bash
set -euo pipefail
root=/home/houj/projects/pianist_roll_capacity_20261028
sha256sum -c <(sed "s|baseline_source.tar.gz|$root/baseline_source.tar.gz|" "$root/baseline_source.tar.gz.sha256")
cp "$root/baseline_source.tar.gz.sha256" "$root/baseline_acceptance/source_archive_sha256.txt"
export PIANIST_ROLL_ACCEPTANCE_DIR="$root/baseline_acceptance"
export PIANIST_EXPECT_ROLL=0
export SBATCH_TIMELIMIT=00:45:00
bash "$root/project/scripts/hpc/submit.sh" --gpus 2 --mem-gb 640 --source-archive "$root/baseline_source.tar.gz" validate tests/test_roll_capacity_acceptance.py -s
