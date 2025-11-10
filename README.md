# rand_core: core random number generation traits

[![crate][crate-image]][crate-link]
[![Docs][docs-image]][docs-link]
![Apache2/MIT licensed][license-image]
![Rust Version][rustc-image]
[![Build Status][build-image]][build-link]

This crate provides a collection of traits used by implementations of Random Number Generation (RNG)
algorithms. Additionally, it includes helper utilities that assist with the implementation
of these traits.

Note that the traits focus solely on the core RNG functionality. Most users should prefer
the [`rand`] crate, which offers more advanced RNG capabilities built on these core traits,
such as sampling from restricted ranges, generating floating-point numbers, list permutations,
and more.

[`rand`]: https://docs.rs/rand

## License

The crate is licensed under either of:

* [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
* [MIT license](http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/rand_core.svg
[crate-link]: https://crates.io/crates/rand_core
[docs-image]: https://docs.rs/rand_core/badge.svg
[docs-link]: https://docs.rs/rand_core
[license-image]: https://img.shields.io/badge/license-Apache2.0/MIT-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.85+-blue.svg
[build-image]: https://github.com/rust-random/rand_core/actions/workflows/test.yml/badge.svg?branch=master
[build-link]: https://github.com/rust-random/rand_core/actions/workflows/test.yml?query=branch:master
