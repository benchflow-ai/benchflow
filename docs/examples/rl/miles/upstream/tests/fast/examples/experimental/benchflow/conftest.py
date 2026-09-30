"""Put the BenchFlow example on sys.path so its modules import by bare name."""

import sys

from . import EXAMPLE_DIR

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
