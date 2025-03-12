from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from polars import Expr, Series
    from polars.datatypes import DataType, DataTypeClass

IntoExprColumn: TypeAlias = "Expr | str | Series"
PolarsDataType: TypeAlias = "DataType | DataTypeClass"
