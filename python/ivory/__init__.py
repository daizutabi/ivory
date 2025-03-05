from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    import polars as pl

    from ivory.typing import IntoExprColumn

LIB = Path(__file__).parent


def abs_i64(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="abs_i64",
        is_elementwise=True,
    )


def pig_latinnify(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="pig_latinnify",
        is_elementwise=True,
    )
