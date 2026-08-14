"""Shared FastAPI dependencies."""

from __future__ import annotations

from types import GeneratorType


def db_dependency():
    """Stable FastAPI dependency wrapper that also keeps test overrides simple.

    直接在函数体内 import 使测试可以 patch ``app.data.get_db`` 生效。
    """
    from app.data import get_db as current_get_db

    value = current_get_db()
    if isinstance(value, GeneratorType):
        yield from value
    else:
        yield value
