"""Self-contained TsenseI piano environment."""
from pathlib import Path

__version__ = "0.3.2"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SF2_PATH = Path(__file__).resolve().parent / "soundfonts" / "default.sf2"
