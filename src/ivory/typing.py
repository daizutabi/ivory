from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polars import Expr, Series
    from polars.datatypes import DataType, DataTypeClass

    type IntoExprColumn = Expr | str | Series
    type PolarsDataType = DataType | DataTypeClass
