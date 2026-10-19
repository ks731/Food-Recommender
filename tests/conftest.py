import sys
from pathlib import Path

# Let tests import the modules in src/ (constraints, ranking, ...)
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))