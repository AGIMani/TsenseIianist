#!/usr/bin/env bash
set -euo pipefail
root=/home/houj/projects/pianist_roll_capacity_20261028
sha256sum -c <(sed "s|source_roll_v1.tar.gz|$root/source_roll_v1.tar.gz|" "$root/source_roll_v1.tar.gz.sha256")
mkdir -p "$root/roll_acceptance_v1"
cp "$root/source_roll_v1.tar.gz.sha256" "$root/roll_acceptance_v1/source_archive_sha256.txt"
export PIANIST_ROLL_ACCEPTANCE_DIR="$root/roll_acceptance_v1"
export PIANIST_EXPECT_ROLL=2
export SBATCH_TIMELIMIT=00:45:00
bash "$root/project/scripts/hpc/submit.sh" --gpus 2 --mem-gb 640 --source-archive "$root/source_roll_v1.tar.gz" validate tests/test_roll_capacity_acceptance.py -s
