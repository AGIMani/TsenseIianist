"""Atomic SAC state snapshots; replay and physics are intentionally not resumed."""
import json
import os
from pathlib import Path
import tempfile

from flax import serialization


def save(path, agent, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialization.msgpack_serialize({
        "agent": serialization.to_state_dict(agent),
        "metadata": json.dumps(metadata, sort_keys=True),
    })
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None and Path(name).exists():
            Path(name).unlink()


def load(path, template, expected):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    metadata = json.loads(payload["metadata"])
    for key in ("contract", "sac_config", "discount"):
        # Normalize tuple/list representations through JSON.
        if metadata[key] != json.loads(json.dumps(expected[key])):
            raise ValueError(f"Checkpoint mismatch: {key}")
    return serialization.from_state_dict(template, payload["agent"]), metadata
