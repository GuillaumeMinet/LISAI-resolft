from __future__ import annotations


def resolve_progress_bar(default: bool, override: bool | None = None, *, stg=None) -> bool:
    """Resolve tqdm preference from CLI override, local config, then caller default."""
    if override is not None:
        return bool(override)

    if stg is None:
        from lisai.config.settings import settings as stg

    local_progress_bar = getattr(stg, "LOCAL_PROGRESS_BAR", None)
    if local_progress_bar is not None:
        return bool(local_progress_bar)

    return bool(default)


__all__ = ["resolve_progress_bar"]
