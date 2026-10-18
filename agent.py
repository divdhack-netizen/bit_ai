"""Compatibility shim: forwards to main.py (the current entry point)."""

from assistant.cli import main

if __name__ == "__main__":
    main()
