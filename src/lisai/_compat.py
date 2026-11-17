"""Compatibility imports for supported Python versions."""

try:  # Python 3.11+
    from importlib.resources.abc import Traversable
except ModuleNotFoundError:  # Python 3.10
    from importlib.abc import Traversable
