"""Root-level entry point - delegates to qms.__main__."""
import sys

from qms.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
