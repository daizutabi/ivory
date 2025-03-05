use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;

#[polars_expr(output_type=Int64)]
fn abs_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.i64()?;
    // NOTE: there's a faster way of implementing `abs_i64`, which we'll
    // cover in section 7.
    let out = ca.apply(|v| v.map(|v| v.abs()));
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
