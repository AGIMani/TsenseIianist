#!/usr/bin/env bash
# Add a bounded 4 GiB local swap reserve; preserve the existing swap.
set -euo pipefail
if ((EUID != 0)); then
  echo 'Run: sudo bash scripts/setup_local_swap.sh' >&2
  exit 1
fi
swap_dir=/var/lib/codex-local-memory
swap_path=$swap_dir/swapfile
swap_bytes=4294967296
if [[ -L "$swap_dir" || -L "$swap_path" ]]; then
  echo 'Refusing a symbolic-link swap destination' >&2
  exit 1
fi
if [[ ! -e "$swap_path" ]]; then
  available=$(df -B1 --output=avail /var/lib | tail -1 | tr -d ' ')
  if ((available < swap_bytes + 4294967296)); then
    echo 'Need at least 8 GiB free: 4 GiB swap plus 4 GiB disk reserve.' >&2
    exit 1
  fi
  install -d -m 700 "$swap_dir"
  install -m 600 /dev/null "$swap_path"
  if ! dd if=/dev/zero of="$swap_path" bs=1M count=4096 status=progress; then
    rm -f -- "$swap_path"
    exit 1
  fi
  mkswap "$swap_path"
elif [[ $(stat -c '%s' "$swap_path") != "$swap_bytes" || $(stat -c '%U:%a' "$swap_path") != 'root:600' ]]; then
  echo 'Existing swap file has unexpected size/owner/permissions; inspect it manually.' >&2
  exit 1
fi
if ! swapon --show=NAME --noheadings | grep -Fxq "$swap_path"; then
  swapon "$swap_path"
fi
if ! awk -v p="$swap_path" '$1 == p && $3 == "swap" { found=1 } END { exit !found }' /etc/fstab; then
  cp -a /etc/fstab "$swap_dir/fstab.before"
  printf '\n%s none swap sw 0 0\n' "$swap_path" >> /etc/fstab
fi
systemctl daemon-reload
swapon --show
free -h
