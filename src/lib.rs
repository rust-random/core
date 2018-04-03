// Copyright 2017-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// https://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Random number generation traits
//! 
//! This crate is mainly of interest to crates publishing implementations of
//! [`RngCore`]. Other users are encouraged to use the [rand] crate instead
//! which re-exports the main traits and error types.
//!
//! [`RngCore`] is the core trait implemented by algorithmic pseudo-random number
//! generators and external random-number sources.
//! 
//! [`SeedableRng`] is an extension trait for construction from fixed seeds and
//! other random number generators.
//! 
//! [`Error`] is provided for error-handling. It is safe to use in `no_std`
//! environments.
//! 
//! The [`impls`] and [`le`] sub-modules include a few small functions to assist
//! implementation of [`RngCore`].
//! 
//! [rand]: https://crates.io/crates/rand
//! [`RngCore`]: trait.RngCore.html
//! [`SeedableRng`]: trait.SeedableRng.html
//! [`Error`]: struct.Error.html
//! [`impls`]: impls/index.html
//! [`le`]: le/index.html

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
       html_favicon_url = "https://www.rust-lang.org/favicon.ico",
       html_root_url = "https://docs.rs/rand_core/0.1")]

#![deny(missing_debug_implementations)]

#![cfg_attr(not(feature="std"), no_std)]
#![cfg_attr(all(feature="alloc", not(feature="std")), feature(alloc))]

#[cfg(feature="std")] extern crate core;
#[cfg(all(feature = "alloc", not(feature="std")))] extern crate alloc;
#[cfg(feature="serde-1")] extern crate serde;
#[cfg(feature="serde-1")] #[macro_use] extern crate serde_derive;


use core::default::Default;
use core::convert::AsMut;

#[cfg(all(feature="alloc", not(feature="std")))] use alloc::boxed::Box;

pub use error::{ErrorKind, Error};


mod error;
pub mod impls;
pub mod le;


/// The core of a random number generator.
/// 
/// This trait encapsulates the low-level functionality common to all
/// generators, and is the "back end", to be implemented by generators.
/// End users should normally use [`Rng`] from the [rand] crate, which is
/// automatically implemented for every type implementing `RngCore`.
/// 
/// Three different methods for generating random data are provided since the
/// optimal implementation of each is dependent on the type of generator. There
/// is no required relationship between the output of each; e.g. many
/// implementations of [`fill_bytes`] consume a whole number of `u32` or `u64`
/// values and drop any remaining unused bytes.
/// 
/// The [`try_fill_bytes`] method is a variant of [`fill_bytes`] allowing error
/// handling; it is not deemed sufficiently useful to add equivalents for
/// [`next_u32`] or [`next_u64`] since the latter methods are almost always used
/// with algorithmic generators (PRNGs), which are normally infallible.
/// 
/// Algorithmic generators implementing [`SeedableRng`] should normally have
/// *portable, reproducible* output, i.e. fix Endianness when converting values
/// to avoid platform differences, and avoid making any changes which affect
/// output (except by communicating that the release has breaking changes).
/// 
/// Typically implementators will implement only one of the methods available
/// in this trait directly, then use the helper functions from the
/// [`rand_core::impls`] module to implement the other methods.
/// 
/// It is recommended that implementations also implement:
/// 
/// - `Debug` with a custom implementation which *does not* print any internal
///   state (at least, [`CryptoRng`]s should not risk leaking state through Debug)
/// - `Serialize` and `Deserialize` (from Serde), preferably making Serde
///   support optional at the crate level in PRNG libs
/// - `Clone` if, and only if, the clone will have identical output to the
///   original (i.e. all deterministic PRNGs but not external generators)
/// - *never* implement `Copy` (accidental copies may cause repeated values)
/// - also *do not* implement `Default`, but instead implement `SeedableRng`
///   thus allowing use of `rand::NewRng` (which is automatically implemented)
/// - `Eq` and `PartialEq` could be implemented, but are probably not useful
/// 
/// # Example
/// 
/// A simple example, obviously not generating very *random* output:
/// 
/// ```rust
/// use rand_core::{RngCore, Error, impls};
/// 
/// struct CountingRng(u64);
/// 
/// impl RngCore for CountingRng {
///     fn next_u32(&mut self) -> u32 {
///         self.next_u64() as u32
///     }
///     
///     fn next_u64(&mut self) -> u64 {
///         self.0 += 1;
///         self.0
///     }
///     
///     fn fill_bytes(&mut self, dest: &mut [u8]) {
///         impls::fill_bytes_via_u64(self, dest)
///     }
///     
///     fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
///         Ok(self.fill_bytes(dest))
///     }
/// }
/// ```
/// 
/// [rand]: https://crates.io/crates/rand
/// [`Rng`]: ../rand/trait.Rng.html
/// [`SeedableRng`]: trait.SeedableRng.html
/// [`rand_core::impls`]: ../rand_core/impls/index.html
/// [`try_fill_bytes`]: trait.RngCore.html#tymethod.try_fill_bytes
/// [`fill_bytes`]: trait.RngCore.html#tymethod.fill_bytes
/// [`next_u32`]: trait.RngCore.html#tymethod.next_u32
/// [`next_u64`]: trait.RngCore.html#tymethod.next_u64
/// [`CryptoRng`]: trait.CryptoRng.html
pub trait RngCore {
    /// Return the next random `u32`.
    ///
    /// RNGs must implement at least one method from this trait directly. In
    /// the case this method is not implemented directly, it can be implemented
    /// using `self.next_u64() as u32` or
    /// [via `fill_bytes`](../rand_core/impls/fn.next_u32_via_fill.html).
    fn next_u32(&mut self) -> u32;

    /// Return the next random `u64`.
    ///
    /// RNGs must implement at least one method from this trait directly. In
    /// the case this method is not implemented directly, it can be implemented
    /// [via `next_u32`](../rand_core/impls/fn.next_u64_via_u32.html) or
    /// [via `fill_bytes`](../rand_core/impls/fn.next_u64_via_fill.html).
    fn next_u64(&mut self) -> u64;

    /// Fill `dest` with random data.
    ///
    /// RNGs must implement at least one method from this trait directly. In
    /// the case this method is not implemented directly, it can be implemented
    /// [via `next_u32`](../rand_core/impls/fn.fill_bytes_via_u32.html) or
    /// [via `next_u64`](../rand_core/impls/fn.fill_bytes_via_u64.html) or
    /// via `try_fill_bytes`; if this generator can fail the implementation
    /// must choose how best to handle errors here (e.g. panic with a
    /// descriptive message or log a warning and retry a few times).
    /// 
    /// This method should guarantee that `dest` is entirely filled
    /// with new data, and may panic if this is impossible
    /// (e.g. reading past the end of a file that is being used as the
    /// source of randomness).
    fn fill_bytes(&mut self, dest: &mut [u8]);

    /// Fill `dest` entirely with random data.
    ///
    /// This is the only method which allows an RNG to report errors while
    /// generating random data thus making this the primary method implemented
    /// by external (true) RNGs (e.g. `OsRng`) which can fail. It may be used
    /// directly to generate keys and to seed (infallible) PRNGs.
    /// 
    /// Other than error handling, this method is identical to [`fill_bytes`];
    /// thus this may be implemented using `Ok(self.fill_bytes(dest))` or
    /// `fill_bytes` may be implemented with
    /// `self.try_fill_bytes(dest).unwrap()` or more specific error handling.
    /// 
    /// [`fill_bytes`]: trait.RngCore.html#method.fill_bytes
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error>;
}

/// A trait for RNGs which do not generate random numbers individually, but in
/// blocks (typically `[u32; N]`). This technique is commonly used by
/// cryptographic RNGs to improve performance.
/// 
/// Usage of this trait is optional, but provides two advantages:
/// implementations only need to concern themselves with generation of the
/// block, not the various [`RngCore`] methods (especially [`fill_bytes`], where the
/// optimal implementations are not trivial), and this allows `ReseedingRng` to
/// perform periodic reseeding with very low overhead.
/// 
/// # Example
/// 
/// ```norun
/// use rand_core::BlockRngCore;
/// use rand_core::impls::BlockRng;
/// 
/// struct MyRngCore;
/// 
/// impl BlockRngCore for MyRngCore {
///     type Results = [u32; 16];
///     
///     fn generate(&mut self, results: &mut Self::Results) {
///         unimplemented!()
///     }
/// }
/// 
/// impl SeedableRng for MyRngCore {
///     type Seed = unimplemented!();
///     fn from_seed(seed: Self::Seed) -> Self {
///         unimplemented!()
///     }
/// }
/// 
/// // optionally, also implement CryptoRng for MyRngCore
/// 
/// // Final RNG.
/// type MyRng = BlockRng<u32, MyRngCore>;
/// ```
/// 
/// [`RngCore`]: trait.RngCore.html
/// [`fill_bytes`]: trait.RngCore.html#tymethod.fill_bytes
pub trait BlockRngCore {
    /// Results element type, e.g. `u32`.
    type Item;
    
    /// Results type. This is the 'block' an RNG implementing `BlockRngCore`
    /// generates, which will usually be an array like `[u32; 16]`.
    type Results: AsRef<[Self::Item]> + Default;

    /// Generate a new block of results.
    fn generate(&mut self, results: &mut Self::Results);
}

/// A marker trait used to indicate that an [`RngCore`] or [`BlockRngCore`]
/// implementation is supposed to be cryptographically secure.
/// 
/// *Cryptographically secure generators*, also known as *CSPRNGs*, should
/// satisfy an additional properties over other generators: given the first
/// *k* bits of an algorithm's output
/// sequence, it should not be possible using polynomial-time algorithms to
/// predict the next bit with probability significantly greater than 50%.
/// 
/// Some generators may satisfy an additional property, however this is not
/// required: if the CSPRNG's state is revealed, it should not be
/// computationally-feasible to reconstruct output prior to this. Some other
/// generators allow backwards-computation and are consided *reversible*.
/// 
/// Note that this trait is provided for guidance only and cannot guarantee
/// suitability for cryptographic applications. In general it should only be
/// implemented for well-reviewed code implementing well-regarded algorithms.
/// 
/// Note also that use of a `CryptoRng` does not protect against other
/// weaknesses such as seeding from a weak entropy source or leaking state.
/// 
/// [`RngCore`]: trait.RngCore.html
/// [`BlockRngCore`]: trait.BlockRngCore.html
pub trait CryptoRng {}

/// A random number generator that can be explicitly seeded.
///
/// This trait encapsulates the low-level functionality common to all
/// pseudo-random number generators (PRNGs, or algorithmic generators).
/// 
/// The [`rand::NewRng`] trait is automatically implemented for every type
/// implementing `SeedableRng`, providing a convenient `new()` method.
/// 
/// [`rand::NewRng`]: ../rand/trait.NewRng.html
pub trait SeedableRng: Sized {
    /// Seed type, which is restricted to types mutably-dereferencable as `u8`
    /// arrays (we recommend `[u8; N]` for some `N`).
    ///
    /// It is recommended to seed PRNGs with a seed of at least circa 100 bits,
    /// which means an array of `[u8; 12]` or greater to avoid picking RNGs with
    /// partially overlapping periods.
    ///
    /// For cryptographic RNG's a seed of 256 bits is recommended, `[u8; 32]`.
    type Seed: Sized + Default + AsMut<[u8]>;

    /// Create a new PRNG using the given seed.
    ///
    /// PRNG implementations are allowed to assume that bits in the seed are
    /// well distributed. That means usually that the number of one and zero
    /// bits are about equal, and values like 0, 1 and (size - 1) are unlikely.
    ///
    /// PRNG implementations are recommended to be reproducible. A PRNG seeded
    /// using this function with a fixed seed should produce the same sequence
    /// of output in the future and on different architectures (with for example
    /// different endianness).
    ///
    /// It is however not required that this function yield the same state as a
    /// reference implementation of the PRNG given equivalent seed; if necessary
    /// another constructor replicating behaviour from a reference
    /// implementation can be added.
    ///
    /// PRNG implementations should make sure `from_seed` never panics. In the
    /// case that some special values (like an all zero seed) are not viable
    /// seeds it is preferable to map these to alternative constant value(s),
    /// for example `0xBAD5EEDu32` or `0x0DDB1A5E5BAD5EEDu64` ("odd biases? bad
    /// seed"). This is assuming only a small number of values must be rejected.
    fn from_seed(seed: Self::Seed) -> Self;

    /// Create a new PRNG seeded from another `Rng`.
    ///
    /// This is the recommended way to initialize PRNGs with fresh entropy. The
    /// [`NewRng`] trait provides a convenient new method based on `from_rng`.
    /// 
    /// Usage of this method is not recommended when reproducibility is required
    /// since implementing PRNGs are not required to fix Endianness and are
    /// allowed to modify implementations in new releases.
    ///
    /// It is important to use a good source of randomness to initialize the
    /// PRNG. Cryptographic PRNG may be rendered insecure when seeded from a
    /// non-cryptographic PRNG or with insufficient entropy.
    /// Many non-cryptographic PRNGs will show statistical bias in their first
    /// results if their seed numbers are small or if there is a simple pattern
    /// between them.
    ///
    /// Prefer to seed from a strong external entropy source like [`OsRng`] or
    /// from a cryptographic PRNG; if creating a new generator for cryptographic
    /// uses you *must* seed from a strong source.
    ///
    /// Seeding a small PRNG from another small PRNG is possible, but
    /// something to be careful with. An extreme example of how this can go
    /// wrong is seeding an Xorshift RNG from another Xorshift RNG, which
    /// will effectively clone the generator. In general seeding from a
    /// generator which is hard to predict is probably okay.
    ///
    /// PRNG implementations are allowed to assume that a good RNG is provided
    /// for seeding, and that it is cryptographically secure when appropriate.
    /// 
    /// [`NewRng`]: ../rand/trait.NewRng.html
    /// [`OsRng`]: ../rand/os/struct.OsRng.html
    fn from_rng<R: RngCore>(mut rng: R) -> Result<Self, Error> {
        let mut seed = Self::Seed::default();
        rng.try_fill_bytes(seed.as_mut())?;
        Ok(Self::from_seed(seed))
    }
}


impl<'a, R: RngCore + ?Sized> RngCore for &'a mut R {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        (**self).next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        (**self).next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        (**self).fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        (**self).try_fill_bytes(dest)
    }
}

#[cfg(feature="alloc")]
impl<R: RngCore + ?Sized> RngCore for Box<R> {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        (**self).next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        (**self).next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        (**self).fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        (**self).try_fill_bytes(dest)
    }
}
