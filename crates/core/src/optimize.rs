use polars::prelude::*;

pub fn linear_fit(
    x: &Float64Chunked,
    y: &Float64Chunked,
) -> PolarsResult<(Option<f64>, Option<f64>)> {
    if x.len() != y.len() {
        polars_bail!(ShapeMismatch: "x and y must have the same length, got x: {}, y: {}", x.len(), y.len());
    }

    if x.is_empty() {
        polars_bail!(ComputeError: "Cannot perform linear regression on empty arrays");
    }

    let valid_mask = x.is_not_null() & y.is_not_null();
    let x = x.filter(&valid_mask)?;
    let y = y.filter(&valid_mask)?;

    if x.len() < 2 {
        return Ok((None, None));
    }

    let x_mean = x.mean().unwrap();
    let y_mean = y.mean().unwrap();

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    x.iter().zip(y.iter()).for_each(|(x, y)| {
        if let (Some(x), Some(y)) = (x, y) {
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }
    });

    if denominator.abs() < f64::EPSILON {
        polars_bail!(ComputeError: "Denominator is zero, cannot compute slope");
    }

    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;

    Ok((Some(slope), Some(intercept)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    #[rstest]
    #[case(vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0], 1.0, 1.0)]
    #[case(vec![1.0, 2.0, 3.0], vec![2.0, 2.0, 2.0], 0.0, 2.0)]
    #[case(vec![1e-2, 2e-2, 3e-2], vec![5.0, 3.0, 1.0], -200.0, 7.0)]
    #[case(vec![1.0, 2.0, 4.0], vec![1.0, 2.0, 3.0], 0.642857, 0.5)]
    #[case(vec![-0.1, -0.5, -2.0], vec![0.3, 0.1, 0.8], -0.318937, 0.123588)]
    #[case(vec![-0.1, 0.0, 1.0, 4.0, 9.0], vec![-0.3, 10.0, 0.1, 2.0, 13.0], 0.946402, 2.329002)]
    fn test_linear_fit(
        #[case] x: Vec<f64>,
        #[case] y: Vec<f64>,
        #[case] expected_slope: f64,
        #[case] expected_intercept: f64,
    ) {
        let x = Float64Chunked::new("x".into(), &x);
        let y = Float64Chunked::new("y".into(), &y);

        let (Some(slope), Some(intercept)) = linear_fit(&x, &y).unwrap() else {
            panic!("Expected Some(slope) and Some(intercept)");
        };
        assert_relative_eq!(slope, expected_slope, epsilon = 1e-6);
        assert_relative_eq!(intercept, expected_intercept, epsilon = 1e-6);
    }

    #[rstest]
    #[case(vec![Some(1.0), Some(2.0), None, Some(3.0)], vec![Some(2.0), Some(3.0), Some(2.0), Some(4.0)], 1.0, 1.0)]
    #[case(vec![Some(1e-2), Some(1e-2), Some(2e-2), Some(3e-2)], vec![Some(5.0), None, Some(3.0), Some(1.0)], -200.0, 7.0)]
    fn test_linear_fit_none(
        #[case] x: Vec<Option<f64>>,
        #[case] y: Vec<Option<f64>>,
        #[case] expected_slope: f64,
        #[case] expected_intercept: f64,
    ) {
        let x = Float64Chunked::new("x".into(), &x);
        let y = Float64Chunked::new("y".into(), &y);

        let (Some(slope), Some(intercept)) = linear_fit(&x, &y).unwrap() else {
            panic!("Expected Some(slope) and Some(intercept)");
        };
        assert_relative_eq!(slope, expected_slope, epsilon = 1e-6);
        assert_relative_eq!(intercept, expected_intercept, epsilon = 1e-6);
    }

    #[test]
    fn test_linear_fit_shape_mismatch() {
        let x = Float64Chunked::new("x".into(), &[1.0, 2.0]);
        let y = Float64Chunked::new("y".into(), &[1.0, 2.0, 3.0]);
        let expected = "x and y must have the same length, got x: 2, y: 3";

        match linear_fit(&x, &y).unwrap_err() {
            PolarsError::ShapeMismatch(msg) => {
                assert_eq!(msg.to_string(), expected)
            }
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_linear_fit_empty() {
        let empty: Vec<f64> = Vec::new();
        let x = Float64Chunked::new("x".into(), &empty);
        let y = Float64Chunked::new("y".into(), &empty);
        let expected = "Cannot perform linear regression on empty arrays";

        match linear_fit(&x, &y).unwrap_err() {
            PolarsError::ComputeError(msg) => {
                assert_eq!(msg.to_string(), expected)
            }
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_linear_fit_not_enough_length() {
        let values = vec![Some(1.0), None];
        let x = Float64Chunked::new("x".into(), &values);
        let y = Float64Chunked::new("y".into(), &values);

        let (None, None) = linear_fit(&x, &y).unwrap() else {
            panic!("Expected None, None");
        };
    }

    #[test]
    fn test_linear_fit_zero_denominator() {
        let x = Float64Chunked::new("x".into(), &[1.0, 1.0]);
        let y = Float64Chunked::new("y".into(), &[2.0, 3.0]);
        let expected = "Denominator is zero, cannot compute slope";

        match linear_fit(&x, &y).unwrap_err() {
            PolarsError::ComputeError(msg) => {
                assert_eq!(msg.to_string(), expected)
            }
            _ => panic!("Unexpected error type"),
        }
    }
}
