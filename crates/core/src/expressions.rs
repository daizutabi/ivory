use num_traits::Signed;
use polars::prelude::*;

pub fn impl_abs_numeric<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Signed,
{
    ca.apply(|opt_v| opt_v.map(|v| v.abs()))
}
