"""Side-effect-free pure identity helper functions and validation utilities.

This module contains no side effects on import and does not open databases.
"""
from typing import Any, Tuple, List, Callable

try:
    from embedders import ModelIdentityError
except ImportError:
    class ModelIdentityError(Exception):
        """Raised when database model identity validation fails (fails closed, no fallback)."""
        pass


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int) and val in (0, 1):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    raise ValueError(f"Noto'g'ri boolean qiymat: {val}")


def _parse_int(val: Any) -> int:
    if isinstance(val, bool):
        raise ValueError(f"Noto'g'ri integer qiymat (bool): {val!r}")
    if isinstance(val, float):
        if not val.is_integer():
            raise ValueError(f"Noto'g'ri integer qiymat (kasr son): {val!r}")
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        s = val.strip()
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
                if f.is_integer():
                    return int(f)
            except (ValueError, OverflowError):
                pass
            raise ValueError(f"Noto'g'ri integer qiymat: {val!r}")
    raise ValueError(f"Noto'g'ri integer qiymat: {val!r}")


IDENTITY_KEYS: List[Tuple[str, Callable[[Any], Any]]] = [
    ("schema_version", _parse_int),
    ("backend", str),
    ("model", str),
    ("model_digest", str),
    ("dimension", _parse_int),
    ("distance", str),
    ("normalization", _parse_bool),
]


def _check_source_internal_conflict(src: dict, name: str = "identity") -> None:
    """Validates every present identity field independently and checks plain-vs-prefixed conflicts."""
    if not isinstance(src, dict):
        return
    for key, cast_type in IDENTITY_KEYS:
        val_plain = src.get(key)
        val_pref = src.get(f"graphify:{key}")
        c1 = None
        c2 = None
        if val_plain is not None:
            try:
                c1 = cast_type(val_plain)
            except (ValueError, TypeError, OverflowError) as e:
                raise ModelIdentityError(
                    f"Model identifikatsiyasi yaroqsiz ({name}): "
                    f"'{key}' qiymatini o'qib bo'lmadi ({val_plain!r})"
                ) from e
        if val_pref is not None:
            try:
                c2 = cast_type(val_pref)
            except (ValueError, TypeError, OverflowError) as e:
                raise ModelIdentityError(
                    f"Model identifikatsiyasi yaroqsiz ({name}): "
                    f"'graphify:{key}' qiymatini o'qib bo'lmadi ({val_pref!r})"
                ) from e
        if val_plain is not None and val_pref is not None:
            differ = False
            try:
                differ = (c1 != c2)
            except Exception:
                differ = True
            if differ:
                raise ModelIdentityError(
                    f"Model identifikatsiyasi nomuvofiq ({name}): "
                    f"'{key}' va 'graphify:{key}' qiymatlari mos kelmadi ({val_plain} != {val_pref})"
                )


def _has_identity_meta(src: dict) -> bool:
    """Checks if a dictionary contains any model identity fields (plain or graphify: prefixed)."""
    if not isinstance(src, dict):
        return False
    for key, _ in IDENTITY_KEYS:
        if key in src or f"graphify:{key}" in src:
            return True
    return False


def _check_identity_conflict(source_a: dict, name_a: str, source_b: dict, name_b: str) -> None:
    """Compares key identity fields between two sources and raises ModelIdentityError on mismatch.
    Validates every present identity field independently in both sources.
    """
    for src, name in ((source_a, name_a), (source_b, name_b)):
        _check_source_internal_conflict(src, name)

    for key, cast_type in IDENTITY_KEYS:
        val_a = source_a.get(key) if source_a.get(key) is not None else source_a.get(f"graphify:{key}")
        val_b = source_b.get(key) if source_b.get(key) is not None else source_b.get(f"graphify:{key}")
        if val_a is not None and val_b is not None:
            try:
                c_a = cast_type(val_a)
                c_b = cast_type(val_b)
            except (ValueError, TypeError, OverflowError) as e:
                raise ModelIdentityError(
                    f"Model identifikatsiyasi nomuvofiq ({name_a} vs {name_b}): "
                    f"{key} qiymatini o'qib bo'lmadi ({val_a} vs {val_b})"
                ) from e
            if c_a != c_b:
                raise ModelIdentityError(
                    f"Model identifikatsiyasi nomuvofiq ({name_a} vs {name_b}): "
                    f"{key} ({val_a} != {val_b})"
                )


def _is_absent_collection_error(e: Exception) -> bool:
    """Checks whether an exception indicates a confirmed absent collection."""
    msg = str(e).lower()
    if "tenant" in msg:
        return False
    try:
        import chromadb.errors
        if isinstance(e, chromadb.errors.NotFoundError):
            return "collection" in msg and ("does not exist" in msg or "not found" in msg)
        if hasattr(chromadb.errors, "InvalidCollectionException") and isinstance(e, chromadb.errors.InvalidCollectionException):
            return True
    except (ImportError, AttributeError):
        pass
    err_cls_name = type(e).__name__
    if err_cls_name == "InvalidCollectionException":
        return True
    if err_cls_name == "NotFoundError":
        return "collection" in msg and ("does not exist" in msg or "not found" in msg)
    if isinstance(e, ValueError):
        return "collection" in msg and ("does not exist" in msg or "not found" in msg)
    return False
