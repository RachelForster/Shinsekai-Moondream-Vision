"""Use the host SDK from the Shinsekai checkout containing this plugin."""

import sys
from pathlib import Path

HOST_ROOT = Path(__file__).resolve().parents[3]
if str(HOST_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_ROOT))
