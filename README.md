[![Github CI](https://github.com/rhaiscript/rhai-sci/actions/workflows/tests.yml/badge.svg)](https://github.com/rhaiscript/rhai-sci/actions)
[![Crates.io](https://img.shields.io/crates/v/rhai-sci.svg)](https://crates.io/crates/rhai-sci)
[![docs.rs](https://img.shields.io/docsrs/rhai-sci/latest?logo=rust)](https://docs.rs/rhai-sci)

# About `rhai-sci`

This crate provides some basic scientific computing utilities for the [`Rhai`](https://rhai.rs/) scripting language,
inspired by languages like MATLAB, Octave, and R. For a complete API reference,
check [the docs](https://docs.rs/rhai-sci).

# Install

To use the latest released version of `rhai-sci`, add this to your `Cargo.toml`:

```toml
rhai-sci = "0.2.2"
```

# Usage

Using this crate is pretty simple! If you just want to evaluate a single line of [`Rhai`](https://rhai.rs/), then you
only need:

```rust
use rhai::INT;
use rhai_sci::eval;
let result = eval::<INT>("argmin([43, 42, -500])").unwrap();
```

If you need to use `rhai-sci` as part of a persistent [`Rhai`](https://rhai.rs/) scripting engine, then do this instead:

```rust
use rhai::{Engine, packages::Package, INT};
use rhai_sci::SciPackage;

// Create a new Rhai engine
let mut engine = Engine::new();

// Add the rhai-sci package to the new engine
engine.register_global_module(SciPackage::new().as_shared_module());

// Now run your code
let value = engine.eval::<INT>("argmin([43, 42, -500])").unwrap();
```

## Matrix helpers

`rhai-sci` provides constructors for row and column vectors and utilities to reorient
`1×N` and `N×1` matrices:

```rust
use rhai::{Array, Dynamic};
use rhai_sci::matrix::RhaiMatrix;

let data: Array = vec![Dynamic::from_int(1), Dynamic::from_int(2)];
let row = RhaiMatrix::row_vector(data.clone());
let column = row.as_column().unwrap();
```

# Features

| Feature    | Default  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `metadata` | Disabled | Enables exporting function metadata and is ___necessary for running doc-tests on Rhai examples___.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `io`       | Enabled  | Enables the [`read_matrix`](#read_matrixfile_path-string---array) function but pulls in several additional dependencies (`polars`, `url`, `temp-file`, `csv-sniffer`, `minreq`).                                                                                                                                                                                                                                                                                                                                      |
| `nalgebra` | Enabled  | Enables several functions ([`regress`](#regressx-array-y-array---map), [`inv`](#invmatrix-array---array), [`mtimes`](#mtimesmatrix1-array-matrix2-array---array), [`horzcat`](#horzcatmatrix1-rhaimatrix-matrix2-rhaimatrix---rhaimatrix), [`vertcat`](#vertcatmatrix1-rhaimatrix-matrix2-rhaimatrix---rhaimatrix), [`repmat`](#repmatmatrix-rhaimatrix-nx-i64-ny-i64---rhaimatrix), [`svd`](#svdmatrix-array---map), [`hessenberg`](#hessenbergmatrix-array---map), and [`qr`](#qrmatrix-array---map)) but brings in the `nalgebra` and `linregress` crates. |
| `rand`     | Enabled  | Enables the [`rand`](#rand) function for generating random FLOAT values and random matrices, but brings in the `rand` crate.                                                                                                                                                                                                                                                                                                                                                                                          |
