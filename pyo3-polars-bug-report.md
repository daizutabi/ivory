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

## Current Resolution
**The issue has been officially fixed!** PR #23918 has been merged and the fix is now available directly from the git repository.

~~Add this module alias before using `polars_expr`:~~

```rust
// ❌ WORKAROUND NO LONGER NEEDED - ISSUE FIXED!
// The official fix in PR #23918 has resolved the arrow module reference issue
// All polars_expr macros now work correctly without any workaround

// OLD WORKAROUND (now removed):
// mod arrow {
//     pub use polars_arrow::ffi;
// }
```

**✅ Simply use polars_expr macros directly - they work perfectly now!**

## Status
- **Reported**: Issue #23902 in the Polars repository
- **Status**: ✅ **FIXED** - PR #23918 merged and applied via git repository
- **Workaround**: ❌ **REMOVED** - No longer needed, official fix is working
- **Impact**: Resolved for all pyo3-polars derive functionality
- **Timeline**: Reported → Diagnosed → Fixed → Merged → Applied within ~12 hours
- **Applied**: August 5, 2025 - Official fix confirmed working in our codebase

## Related Issues
- [polars#23902](https://github.com/pola-rs/polars/issues/23902) - Main issue tracker
- [polars#23893](https://github.com/pola-rs/polars/issues/23893) - Related compilation issue
