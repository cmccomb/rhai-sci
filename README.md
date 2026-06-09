[![Github CI](https://github.com/rhaiscript/rhai-sci/actions/workflows/tests.yml/badge.svg)](https://github.com/rhaiscript/rhai-sci/actions)
[![Crates.io](https://img.shields.io/crates/v/rhai-sci.svg)](https://crates.io/crates/rhai-sci)
[![docs.rs](https://img.shields.io/docsrs/rhai-sci/latest?logo=rust)](https://docs.rs/rhai-sci)

# rhai-sci

## What & Why

`rhai-sci` adds basic scientific computing utilities to the [Rhai](https://rhai.rs/) scripting language. It is inspired by tools such as MATLAB, Octave, and R.

## Quickstart

Add the crate to your `Cargo.toml`:

```toml
rhai-sci = "0.2.3"
```

Evaluate a single line of Rhai code:

```rust
use rhai::INT;
use rhai_sci::eval;

let result = eval::<INT>("argmin([43, 42, -500])").unwrap();
```

## Examples

Integrate the package with a persistent Rhai engine:

```rust
use rhai::{packages::Package, Engine, INT};
use rhai_sci::SciPackage;

let mut engine = Engine::new();
engine.register_global_module(SciPackage::new().as_shared_module());

let value = engine.eval::<INT>("argmin([43, 42, -500])").unwrap();
```

See the `examples` directory for more:

- `matrix_inversion.rhai` demonstrates matrix inversion.
- `download_and_regress.rhai` fetches data and performs linear regression.
- `projectile_motion.rhai` uses trigonometry and array utilities to simulate a projectile trajectory.

## Matrix and Vector Conventions

Rhai arrays remain the storage model. Use constructors when shape matters:

```typescript
let values = [1, 2, 3];        // plain Rhai list
let x = vec([1, 2, 3]);        // column vector: [[1], [2], [3]]
let c = col([1, 2, 3]);        // column vector: [[1], [2], [3]]
let r = row([1, 2, 3]);        // row vector: [[1, 2, 3]]
let A = mat([[1, 2], [3, 4]]); // validated matrix
```

For compact matrix literals, use strings with whitespace, commas, and semicolons:

```typescript
let A = mat("1 2; 3 4");
let B = M("1, 2; 3, 4");
let r = R("1 2 3");
let c = C("1; 2; 3");
```

The short aliases keep linear algebra scripts readable:

```typescript
let A = mat("1 2; 3 4");
let x = col([5, 6]);
let y = row([7, 8]);

let z = dot(A, x);
let z2 = A.dot(x);
let At = T(A);
let C = hcat(A, x);
let D = vcat(A, y);
```

### Features

- **metadata** *(disabled)*: export function metadata; required for running doc-tests on Rhai examples.
- **io** *(enabled)*: provides `read_matrix` but pulls in `polars`, `url`, `temp-file`, `csv-sniffer`, and `minreq`.
- **nalgebra** *(enabled)*: enables matrix functions such as `regress`, `inv`, `mtimes`, `horzcat`, `vertcat`, `repmat`, `svd`, `hessenberg`, and `qr` via the `nalgebra` and `linregress` crates.
- **rand** *(enabled)*: adds the `rand` function for generating random values and matrices using the `rand` crate.

## CLI/API reference

The full API is documented on [docs.rs](https://docs.rs/rhai-sci).

## Development

```bash
cargo fmt --all -- --check
cargo test --no-default-features --features rand,nalgebra
```

## Troubleshooting

- Building with the `io` feature enabled pulls in heavy dependencies. Disable default features and enable only what you need if builds are slow.

## License

Licensed under either of

- [MIT license](LICENSE-MIT.txt)
- [Apache License, Version 2.0](LICENSE-APACHE.txt)

at your option.
