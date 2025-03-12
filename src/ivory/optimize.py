from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars import Expr

    from .typing import IntoExprColumn


def linear_fit(x: IntoExprColumn, y: IntoExprColumn) -> Expr:
    """Perform linear regression on the input series.

    This function fits a linear model to the input series and
    returns the slope and intercept.

    Args:
        x (IntoExprColumn): The independent variable.
        y (IntoExprColumn): The dependent variable.

    Returns:
        Expr: A polars expression that evaluates to a struct
        with the slope and intercept.
    """
    return register_plugin_function(
        args=[x, y],
        plugin_path=Path(__file__).parent,
        function_name="linear_fit",
        is_elementwise=False,
        returns_scalar=True,
    )


def linear_transform_fit(x: IntoExprColumn, y: IntoExprColumn) -> Expr:
    """Perform linear regression on the input series.

    This function fits a linear model to the input series and
    returns the scale and offset.

    Args:
        x (IntoExprColumn): The independent variable.
        y (IntoExprColumn): The dependent variable.

    Returns:
        Expr: A polars expression that evaluates to a struct
        with the scale and offset.
    """
    return register_plugin_function(
        args=[x, y],
        plugin_path=Path(__file__).parent,
        function_name="linear_transform_fit",
        is_elementwise=False,
        returns_scalar=True,
    )
