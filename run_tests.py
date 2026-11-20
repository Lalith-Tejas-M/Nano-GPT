"""Convenience test runner that automatically configures Windows environment isolation."""
import os
import sys

# Isolate from global setuptools entrypoints that may load blocked third-party C-extensions on Windows
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

import pytest

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["-v"]
    sys.exit(pytest.main(args))
