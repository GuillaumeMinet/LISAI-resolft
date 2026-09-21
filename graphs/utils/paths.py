from pathlib import Path


GRAPHS_DIR = Path(__file__).resolve().parents[1]
SAVED_GRAPHS_DIR = GRAPHS_DIR / "saved_graphs"


def get_saved_graphs_dir() -> Path:
    """ Returns shared saving path: LISAI/graphs/saved_graphs"""
    SAVED_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    return SAVED_GRAPHS_DIR