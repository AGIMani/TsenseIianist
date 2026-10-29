#!/usr/bin/env bash
set -euo pipefail
root=/home/houj/projects/pianist_palm_ranges_20261028
project=/home/houj/projects/pianist_roll_capacity_20261028/project
sha256sum -c <(sed "s|source_palm_ranges_v1.tar.gz|$root/source_palm_ranges_v1.tar.gz|" "$root/source_palm_ranges_v1.tar.gz.sha256")
cp "$root/source_palm_ranges_v1.tar.gz.sha256" "$root/acceptance/source_archive_sha256.txt"
export PIANIST_ROLL_ACCEPTANCE_DIR="$root/acceptance"
export SBATCH_TIMELIMIT=00:15:00
bash "$project/scripts/hpc/submit.sh" --gpus 1 --mem-gb 32 --source-archive "$root/source_palm_ranges_v1.tar.gz" validate tests/test_palm_ranges_acceptance.py -s
