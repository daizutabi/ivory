# Bug Report: pyo3-polars 0.23 - polars_expr macro generates unresolved arrow module reference

**Note: This issue has been reported and is being tracked at [polars#23902](https://github.com/pola-rs/polars/issues/23902)**

## Summary
This document serves as our internal documentation of the issue and workaround. The problem has been officially reported to the Polars team and is under investigation.

## Bug Description
The `polars_expr` procedural macro generates an unresolved reference to `arrow` module, causing compilation failures when using pyo3-polars with the `derive` feature. **This affects ALL uses of the `polars_expr` macro**, including both `output_type` and `output_type_func` variants.

## Environment
- **pyo3-polars**: 0.23.0 (with `derive` feature)
- **polars**: 0.50.0
- **Rust**: 1.82+ (latest stable)
- **OS**: Ubuntu 24.04.2 LTS (but affects all platforms)

## Current Workaround
Add this module alias before using `polars_expr`:

```rust
// Workaround for pyo3-polars bug: polars_expr macro generates unresolved arrow module reference
// The macro should reference polars_arrow::ffi but instead generates arrow::ffi
// TODO: Remove when https://github.com/pola-rs/polars/issues/23902 is fixed
// This affects ALL polars_expr usage, not just output_type_func
mod arrow {
    pub use polars_arrow::ffi;
}
```

## Status
- **Reported**: Issue #23902 in the Polars repository
- **Status**: Under investigation by maintainers
- **Workaround**: Available and functional
- **Impact**: Affects all pyo3-polars derive functionality

## Related Issues
- [polars#23902](https://github.com/pola-rs/polars/issues/23902) - Main issue tracker
- [polars#23893](https://github.com/pola-rs/polars/issues/23893) - Related compilation issue
