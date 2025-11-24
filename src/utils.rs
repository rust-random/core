//! Utilties to aid trait implementations
//!
//! ## Portability
//!
//! For cross-platform reproducibility, Little-Endian order (least-significant
//! part first) has been chosen as the standard for inter-type conversion.
//! For example, [`next_u64_via_u32`] generates two `u32` values `x, y`,
//! then outputs `(y << 32) | x`.
//!
//! Byte-swapping (like the std `to_le` functions) is only needed to convert
//! to/from byte sequences, and since its purpose is reproducibility,
//! non-reproducible sources (e.g. `OsRng`) need not bother with it.
//!
//! ## Implementing [`SeedableRng`]
//!
//! In many cases, [`SeedableRng::Seed`] must be converted to `[u32]` or `[u64]`.
//! We provide the [`read_words`] helper function for this. The examples below
//! demonstrate how it can be used in practice.
//!
//! [`SeedableRng`]: crate::SeedableRng
//! [`SeedableRng::Seed`]: crate::SeedableRng::Seed
//!
//! ## Implementing [`RngCore`]
//!
//! Usually an implementation of [`RngCore`] will implement one of the three methods
//! over its internal source, while remaining methods are implemented on top of it.
//! We consider the following cases.
//!
//! ### Word generator (`u32`)
//!
//! We demonstrate a simple "step RNG":
//! ```
//! use rand_core::{RngCore, SeedableRng, utils};
//!
//! pub struct Step32Rng {
//!     state: u32
//! }
//!
//! impl SeedableRng for Step32Rng {
//!     type Seed = [u8; 4];
//!
//!     #[inline]
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         // Always use little-endian byte order to ensure portable results
//!         let state = u32::from_le_bytes(seed);
//!         Self { state }
//!     }
//! }
//!
//! impl RngCore for Step32Rng {
//!     #[inline]
//!     fn next_u32(&mut self) -> u32 {
//!         let val = self.state;
//!         self.state = val + 1;
//!         val
//!     }
//!
//!     #[inline]
//!     fn next_u64(&mut self) -> u64 {
//!         utils::next_u64_via_u32(self)
//!     }
//!
//!     #[inline]
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         utils::fill_bytes_via_next_word(dst, || self.next_u32());
//!     }
//! }
//!
//! # let mut rng = Step32Rng::seed_from_u64(42);
//! # assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! # assert_eq!(rng.next_u64(), 0x7ba1_8fa6_7ba1_8fa5);
//! # let mut buf = [0u8; 5];
//! # rng.fill_bytes(&mut buf);
//! # assert_eq!(buf, [0xa7, 0x8f, 0xa1, 0x7b, 0xa8]);
//! ```
//!
//! Use of `#[inline]` allows function inlining across crate boundaries, which
//! may aid optimization. We use it here since these methods are small,
//! non-generic and may be performance sensitive.
//! See also [When to `#[inline]`](https://std-dev-guide.rust-lang.org/policy/inline.html).
//!
//! ### Word generator (`u64`)
//!
//! This example is similar:
//! ```
//! # use rand_core::{RngCore, SeedableRng, utils};
//! pub struct Step64Rng(u64);
//! # impl SeedableRng for Step64Rng {
//! #     type Seed = [u8; 8];
//! #
//! #     #[inline]
//! #     fn from_seed(seed: Self::Seed) -> Self {
//! #         Self(u64::from_le_bytes(seed))
//! #     }
//! # }
//!
//! impl RngCore for Step64Rng {
//!     #[inline]
//!     fn next_u32(&mut self) -> u32 {
//!         self.next_u64() as u32
//!     }
//!
//!     #[inline]
//!     fn next_u64(&mut self) -> u64 {
//!         // ...
//!         # let val = self.0;
//!         # self.0 = val + 1;
//!         # val
//!     }
//!
//!     #[inline]
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         utils::fill_bytes_via_next_word(dst, || self.next_u64());
//!     }
//! }
//!
//! # let mut rng = Step64Rng::seed_from_u64(42);
//! # assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! # assert_eq!(rng.next_u64(), 0x0a3d_3258_7ba1_8fa5);
//! # let mut buf = [0u8; 5];
//! # rng.fill_bytes(&mut buf);
//! # assert_eq!(buf, [0xa6, 0x8f, 0xa1, 0x7b, 0x58]);
//! ```
//!
//! ### Byte stream sources
//!
//! Sources that yield byte streams (for example, file readers) can be adapted
//! to [`RngCore`] or [`TryRngCore`] using only `from_le_bytes`:
//! ```
//! use rand_core::RngCore;
//!
//! pub struct FillRng {
//!     // ...
//!     # state: u8,
//! }
//!
//! impl RngCore for FillRng {
//!     #[inline]
//!     fn next_u32(&mut self) -> u32 {
//!         let mut buf = [0; 4];
//!         self.fill_bytes(&mut buf);
//!         u32::from_le_bytes(buf)
//!     }
//!
//!     #[inline]
//!     fn next_u64(&mut self) -> u64 {
//!         let mut buf = [0; 8];
//!         self.fill_bytes(&mut buf);
//!         u64::from_le_bytes(buf)
//!     }
//!
//!     #[inline]
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         // ...
//!         # for byte in dst {
//!         #     let val = self.state;
//!         #     self.state = val + 1;
//!         #     *byte = val;
//!         # }
//!     }
//! }
//!
//! # let mut rng = FillRng { state: 0 };
//! # assert_eq!(rng.next_u32(), 0x03_020100);
//! # assert_eq!(rng.next_u64(), 0x0b0a_0908_0706_0504);
//! # let mut buf = [0u8; 5];
//! # rng.fill_bytes(&mut buf);
//! # assert_eq!(buf, [0x0c, 0x0d, 0x0e, 0x0f, 0x10]);
//! ```
//!
//! Note that you can use `from_ne_bytes` instead of `from_le_bytes`
//! if your `fill_bytes` implementation is not reproducible.
//!
//! [`TryRngCore`]: crate::TryRngCore

use crate::RngCore;

/// Sealed trait implemented for `u32` and `u64`.
pub trait Word: crate::sealed::Sealed {}

impl Word for u32 {}
impl Word for u64 {}

/// Implement `next_u64` via `next_u32` using little-endian order.
#[inline(always)]
pub fn next_u64_via_u32<R: RngCore + ?Sized>(rng: &mut R) -> u64 {
    // Use LE; we explicitly generate one value before the next.
    let x = u64::from(rng.next_u32());
    let y = u64::from(rng.next_u32());
    (y << 32) | x
}

/// Implement `fill_bytes` via `next_u64` using little-endian order.
#[inline]
pub fn fill_bytes_via_next_word<W: Word>(dst: &mut [u8], mut next_word: impl FnMut() -> W) {
    let mut chunks = dst.chunks_exact_mut(size_of::<W>());
    for chunk in &mut chunks {
        let val = next_word();
        chunk.copy_from_slice(val.to_le_bytes().as_ref());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
        let val = next_word().to_le_bytes();
        rem.copy_from_slice(&val.as_ref()[..rem.len()]);
    }
}

/// Reads array of words from byte slice `src` using little endian order.
///
/// # Panics
/// If `size_of_val(src) != size_of::<[W; N]>()`.
#[inline(always)]
pub fn read_words<W: Word, const N: usize>(src: &[u8]) -> [W; N] {
    assert_eq!(size_of_val(src), size_of::<[W; N]>());
    let mut dst = [W::from_usize(0); N];
    let chunks = src.chunks_exact(size_of::<W>());
    for (out, chunk) in dst.iter_mut().zip(chunks) {
        let Ok(bytes) = chunk.try_into() else {
            unreachable!()
        };
        *out = W::from_le_bytes(bytes);
    }
    dst
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_read() {
        let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let buf: [u32; 4] = read_words(&bytes);
        assert_eq!(buf[0], 0x04030201);
        assert_eq!(buf[3], 0x100F0E0D);

        let buf: [u32; 3] = read_words(&bytes[1..13]); // unaligned
        assert_eq!(buf[0], 0x05040302);
        assert_eq!(buf[2], 0x0D0C0B0A);

        let buf: [u64; 2] = read_words(&bytes);
        assert_eq!(buf[0], 0x0807060504030201);
        assert_eq!(buf[1], 0x100F0E0D0C0B0A09);

        let buf: [u64; 1] = read_words(&bytes[7..15]); // unaligned
        assert_eq!(buf[0], 0x0F0E0D0C0B0A0908);
    }
}
