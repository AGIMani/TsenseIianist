"""Persistent machine connection settings, separate from disposable logs."""
from pathlib import Path


def connection_config(root, backend, *, migrate=True):
    if backend not in ('remote', 'local'):
        raise ValueError('Unknown editor backend')
    root = Path(root)
    destination = root/'configs/fingering_editor'/f'{backend}.json'
    legacy = root/'logs/fingering_editor'/f'{backend}.json'
    if migrate and not destination.exists() and legacy.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = legacy.read_bytes()
        # Exclusive creation preserves an already configured destination.
        try:
            with destination.open('xb') as output:
                output.write(raw)
        except FileExistsError:
            return destination
        if destination.read_bytes() != raw:
            raise OSError('Connection configuration migration verification failed')
        legacy.unlink()
    return destination
