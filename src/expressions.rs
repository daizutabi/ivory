use num_traits::Signed;
use polars::prelude::arity::broadcast_binary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use std::fmt::Write;

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

#[polars_expr(output_type=Int64)]
fn abs_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.i64()?;
    // NOTE: there's a faster way of implementing `abs_i64`, which we'll
    // cover in section 7.
    let out = ca.apply(|v| v.map(|v| v.abs()));
    Ok(out.into_series())
}

fn impl_abs_numeric<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Signed,
{
    // NOTE: there's a faster way of implementing `abs`, which we'll
    // cover in section 7.
    ca.apply(|opt_v| opt_v.map(|v| v.abs()))
}

#[polars_expr(output_type_func=same_output_type)]
fn abs_numeric(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::Int32 => Ok(impl_abs_numeric(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(impl_abs_numeric(s.i64().unwrap()).into_series()),
        DataType::Float32 => Ok(impl_abs_numeric(s.f32().unwrap()).into_series()),
        DataType::Float64 => Ok(impl_abs_numeric(s.f64().unwrap()).into_series()),
        dtype => {
            polars_bail!(InvalidOperation:format!("dtype {dtype} not \
            supported for abs_numeric, expected Int32, Int64, Float32, Float64."))
        }
    }
}

#[polars_expr(output_type=Int64)]
fn sum_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let left = inputs[0].i64()?;
    let right = inputs[1].i64()?;
    // Note: there's a faster way of summing two columns, see
    // section 7.
    let out: Int64Chunked =
        broadcast_binary_elementwise(left, right, |left, right| match (left, right) {
            (Some(left), Some(right)) => Some(left + right),
            _ => None,
        });
    Ok(out.into_series())
}

#[polars_expr(output_type_func=same_output_type)]
fn cum_sum(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.i64()?;
    let out: Int64Chunked = ca
        .iter()
        .scan(0_i64, |state, x| match x {
            Some(x) => {
                *state += x;
                Some(Some(*state))
            }
            None => Some(None),
        })
        .collect_trusted();
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out = ca.apply_into_string_amortized(|value, output| {
        if let Some(first_char) = value.chars().next() {
            write!(output, "{}{}ay", &value[1..], first_char).unwrap()
        }
    });
    Ok(out.into_series())
}
