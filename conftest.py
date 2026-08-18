import os
import sys

# Ensure src/ is on sys.path so tests can import src.api
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def pytest_ignore_collect(path):
    """Ignore known interactive/manual test scripts that block CI."""
    interactive_files = {
        "test_arslm_final.py",
        "test_arslm_termux.py",
        "test_arslm_cpu.py",
        "test_arslm.py",
        "test_arslm_lora.py",
    }
    if path.basename in interactive_files:
        return True
