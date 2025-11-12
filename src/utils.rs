//! Helper utilities.
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
//! # Implementing [`SeedableRng`]
//!
//! In many cases, [`SeedableRng::Seed`] must be converted to `[u32]` or `[u64]`.
//! We provide [`read_u32_into`] and [`read_u64_into`] helpers for this.
//!
//! [`SeedableRng`]: crate::SeedableRng
//! [`SeedableRng::Seed`]: crate::SeedableRng::Seed
//!
//! # Implementing [`RngCore`]
//!
//! Usually an implementation of [`RngCore`] will implement one of the three methods
//! over its internal source, while remaining methods are implemented on top of it.
//!
//! Additionally, some RNGs generate blocks of data. In that case the implementations have to
//! handle buffering of the generated block. If an implementation supports SIMD-based optimizations,
//! i.e. if optimal block size depends on available target features, we reccomend to always
//! generate the biggest supported block size.
//!
//! See the examples below which demonstrate how functions in this module can be used to implement
//! `RngCore` for common RNG algorithm classes.
//!
//! WARNING: the RNG implementations below are provided for demonstation purposes only and
//! should not be used in practice!
//!
//! ## Fill-based RNG
//!
//! ```
//! use rand_core::RngCore;
//!
//! pub struct FillRng(u8);
//!
//! impl RngCore for FillRng {
//!     fn next_u32(&mut self) -> u32 {
//!         let mut buf = [0; 4];
//!         self.fill_bytes(&mut buf);
//!         u32::from_le_bytes(buf)
//!     }
//!
//!     fn next_u64(&mut self) -> u64 {
//!         let mut buf = [0; 8];
//!         self.fill_bytes(&mut buf);
//!         u64::from_le_bytes(buf)
//!     }
//!
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         for byte in dst {
//!             let val = self.0;
//!             self.0 = val + 1;
//!             *byte = val;
//!         }
//!     }
//! }
//!
//! let mut rng = FillRng(0);
//!
//! assert_eq!(rng.next_u32(), 0x03_020100);
//! assert_eq!(rng.next_u64(), 0x0b0a_0908_0706_0504);
//! let mut buf = [0u8; 5];
//! rng.fill_bytes(&mut buf);
//! assert_eq!(buf, [0x0c, 0x0d, 0x0e, 0x0f, 0x10]);
//! ```
//!
//! Note that you can use `from_ne_bytes` instead of `from_le_bytes`
//! if your `fill_bytes` implementation is not reproducible.
//!
//! ## Single 32-bit value RNG
//!
//! ```
//! use rand_core::{RngCore, utils};
//!
//! pub struct Step32Rng(u32);
//!
//! impl RngCore for Step32Rng {
//!     fn next_u32(&mut self) -> u32 {
//!         let val = self.0;
//!         self.0 = val + 1;
//!         val
//!     }
//!
//!     fn next_u64(&mut self) -> u64 {
//!         utils::next_u64_via_u32(self)
//!     }
//!
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         utils::fill_bytes_via_next(self, dst);
//!     }
//! }
//!
//! let mut rng = Step32Rng(0);
//!
//! assert_eq!(rng.next_u32(), 0);
//! assert_eq!(rng.next_u64(), 0x0000_0002_0000_0001);
//! let mut buf = [0u8; 5];
//! rng.fill_bytes(&mut buf);
//! assert_eq!(buf, [3, 0, 0, 0, 4]);
//! ```
//!
//! ## Single 64-bit value RNG
//!
//! ```
//! use rand_core::{RngCore, utils};
//!
//! pub struct Step64Rng(u64);
//!
//! impl RngCore for Step64Rng {
//!     fn next_u32(&mut self) -> u32 {
//!         self.next_u64() as u32
//!     }
//!
//!     fn next_u64(&mut self) -> u64 {
//!         let val = self.0;
//!         self.0 = val + 1;
//!         val
//!     }
//!
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         utils::fill_bytes_via_next(self, dst);
//!     }
//! }
//!
//! let mut rng = Step64Rng(0);
//!
//! assert_eq!(rng.next_u32(), 0);
//! assert_eq!(rng.next_u64(), 1);
//! let mut buf = [0u8; 5];
//! rng.fill_bytes(&mut buf);
//! assert_eq!(buf, [2, 0, 0, 0, 0]);
//! ```
//!
//! ## 32-bit block RNG
//!
//! ```
//! use rand_core::{RngCore, SeedableRng, utils};
//!
//! struct Block32RngCore([u32; 8]);
//!
//! impl Block32RngCore {
//!     fn next_block(&mut self) -> [u32; 8] {
//!         let val = self.0;
//!         self.0 = val.map(|v| v + 1);
//!         val
//!     }
//! }
//!
//! pub struct Block32Rng {
//!     core: Block32RngCore,
//!     buffer: [u32; 8],
//! }
//!
//! impl SeedableRng for Block32Rng {
//!     type Seed = [u8; 32];
//!
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         let mut core_state = [0u32; 8];
//!         utils::read_u32_into(&seed, &mut core_state);
//!         Self {
//!             core: Block32RngCore(core_state),
//!             buffer: utils::new_u32_buffer(),
//!         }
//!     }
//! }
//!
//! impl RngCore for Block32Rng {
//!     fn next_u32(&mut self) -> u32 {
//!         utils::next_u32_from_block(&mut self.buffer, || self.core.next_block())
//!     }
//!
//!     fn next_u64(&mut self) -> u64 {
//!         utils::next_u64_via_u32(self)
//!     }
//!
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         utils::fill_bytes_via_next(self, dst);
//!     }
//! }
//!
//! let mut rng = Block32Rng::seed_from_u64(42);
//!
//! assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! assert_eq!(rng.next_u64(), 0xcca1_b8ea_0a3d_3258);
//! let mut buf = [0u8; 5];
//! rng.fill_bytes(&mut buf);
//! assert_eq!(buf, [0x69, 0x01, 0x14, 0xb8, 0x2b]);
//! ```
//!
//! ## 64-bit block RNG
//!
//! ```
//! use rand_core::{RngCore, SeedableRng, utils};
//!
//! struct Block64RngCore([u64; 4]);
//!
//! impl Block64RngCore {
//!     fn next_block(&mut self) -> [u64; 4] {
//!         let val = self.0;
//!         self.0 = val.map(|v| v + 1);
//!         val
//!     }
//! }
//!
//! pub struct Block64Rng {
//!     core: Block64RngCore,
//!     buffer: [u64; 4],
//! }
//!
//! impl SeedableRng for Block64Rng {
//!     type Seed = [u8; 32];
//!
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         let mut core_state = [0u64; 4];
//!         utils::read_u64_into(&seed, &mut core_state);
//!         Self {
//!             core: Block64RngCore(core_state),
//!             buffer: utils::new_u64_buffer(),
//!         }
//!     }
//! }
//!
//! impl RngCore for Block64Rng {
//!     fn next_u32(&mut self) -> u32 {
//!         self.next_u64() as u32
//!     }
//!
//!     fn next_u64(&mut self) -> u64 {
//!         utils::next_u64_from_block(&mut self.buffer, || self.core.next_block())
//!     }
//!
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         utils::fill_bytes_via_next(self, dst);
//!     }
//! }
//!
//! let mut rng = Block64Rng::seed_from_u64(42);
//!
//! assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! assert_eq!(rng.next_u64(), 0xb814_0169_cca1_b8ea);
//! let mut buf = [0u8; 5];
//! rng.fill_bytes(&mut buf);
//! assert_eq!(buf, [0x2b, 0x8c, 0xc8, 0x75, 0x18]);
//! ```

use crate::RngCore;

/// Implement `next_u64` via `next_u32`, little-endian order.
pub fn next_u64_via_u32<R: RngCore + ?Sized>(rng: &mut R) -> u64 {
    // Use LE; we explicitly generate one value before the next.
    let x = u64::from(rng.next_u32());
    let y = u64::from(rng.next_u32());
    (y << 32) | x
}

/// Implement `fill_bytes` via `next_u64` and `next_u32`, little-endian order.
///
/// The fastest way to fill a slice is usually to work as long as possible with
/// integers. That is why this method mostly uses `next_u64`, and only when
/// there are 4 or less bytes remaining at the end of the slice it uses
/// `next_u32` once.
pub fn fill_bytes_via_next<R: RngCore + ?Sized>(rng: &mut R, dest: &mut [u8]) {
    let mut left = dest;
    while left.len() >= 8 {
        let (l, r) = { left }.split_at_mut(8);
        left = r;
        let chunk: [u8; 8] = rng.next_u64().to_le_bytes();
        l.copy_from_slice(&chunk);
    }
    let n = left.len();
    if n > 4 {
        let chunk: [u8; 8] = rng.next_u64().to_le_bytes();
        left.copy_from_slice(&chunk[..n]);
    } else if n > 0 {
        let chunk: [u8; 4] = rng.next_u32().to_le_bytes();
        left.copy_from_slice(&chunk[..n]);
    }
}

/// Fills `dst: &mut [u32]` from `src`.
///
/// Reads use Little-Endian byte order, allowing portable reproduction of `dst`
/// from a byte slice.
///
/// # Panics
///
/// If `src.len() != 4 * dst.len()`.
#[inline]
#[track_caller]
pub fn read_u32_into(src: &[u8], dst: &mut [u32]) {
    assert_eq!(size_of_val(src), size_of_val(dst));
    for (out, chunk) in dst.iter_mut().zip(src.chunks_exact(4)) {
        *out = u32::from_le_bytes(chunk.try_into().unwrap());
    }
}

/// Fills `dst: &mut [u64]` from `src`.
///
/// # Panics
///
/// If `src.len() != 8 * dst.len()`.
#[inline]
#[track_caller]
pub fn read_u64_into(src: &[u8], dst: &mut [u64]) {
    assert_eq!(size_of_val(src), size_of_val(dst));
    for (out, chunk) in dst.iter_mut().zip(src.chunks_exact(8)) {
        *out = u64::from_le_bytes(chunk.try_into().unwrap());
    }
}

/// Create new 32-bit block buffer.
pub fn new_u32_buffer<const N: usize>() -> [u32; N] {
    assert!(N > 1);
    let mut res = [0u32; N];
    res[0] = N.try_into().unwrap();
    res
}

/// Generate `u32` from block.
pub fn next_u32_from_block<const N: usize>(
    buf: &mut [u32; N],
    mut generate_block: impl FnMut() -> [u32; N],
) -> u32 {
    let pos = buf[0] as usize;
    match buf.get(pos) {
        Some(&val) => {
            buf[0] += 1;
            val
        }
        None => {
            *buf = generate_block();
            core::mem::replace(&mut buf[0], 1)
        }
    }
}

/// Create new 64-bit block buffer.
pub fn new_u64_buffer<const N: usize>() -> [u64; N] {
    assert!(N > 1);
    let mut res = [0u64; N];
    res[0] = N.try_into().unwrap();
    res
}

/// Generate `u64` from block.
pub fn next_u64_from_block<const N: usize>(
    buf: &mut [u64; N],
    mut generate_block: impl FnMut() -> [u64; N],
) -> u64 {
    let pos = buf[0] as usize;
    match buf.get(pos) {
        Some(&val) => {
            buf[0] += 1;
            val
        }
        None => {
            *buf = generate_block();
            core::mem::replace(&mut buf[0], 1)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    pub(crate) trait Observable: Copy {
        type Bytes: Sized + AsRef<[u8]>;
        fn to_le_bytes(self) -> Self::Bytes;
    }
    impl Observable for u32 {
        type Bytes = [u8; 4];

        fn to_le_bytes(self) -> Self::Bytes {
            Self::to_le_bytes(self)
        }
    }
    impl Observable for u64 {
        type Bytes = [u8; 8];

        fn to_le_bytes(self) -> Self::Bytes {
            Self::to_le_bytes(self)
        }
    }

    /// Fill dest from src
    ///
    /// Returns `(n, byte_len)`. `src[..n]` is consumed,
    /// `dest[..byte_len]` is filled. `src[n..]` and `dest[byte_len..]` are left
    /// unaltered.
    pub(crate) fn fill_via_chunks<T: Observable>(src: &[T], dest: &mut [u8]) -> (usize, usize) {
        let size = core::mem::size_of::<T>();

        // Always use little endian for portability of results.

        let mut dest = dest.chunks_exact_mut(size);
        let mut src = src.iter();

        let zipped = dest.by_ref().zip(src.by_ref());
        let num_chunks = zipped.len();
        zipped.for_each(|(dest, src)| dest.copy_from_slice(src.to_le_bytes().as_ref()));

        let byte_len = num_chunks * size;
        if let Some(src) = src.next() {
            // We have consumed all full chunks of dest, but not src.
            let dest = dest.into_remainder();
            let n = dest.len();
            if n > 0 {
                dest.copy_from_slice(&src.to_le_bytes().as_ref()[..n]);
                return (num_chunks + 1, byte_len + n);
            }
        }
        (num_chunks, byte_len)
    }

    #[test]
    fn test_fill_via_u32_chunks() {
        let src_orig = [1u32, 2, 3];

        let src = src_orig;
        let mut dst = [0u8; 11];
        assert_eq!(fill_via_chunks(&src, &mut dst), (3, 11));
        assert_eq!(dst, [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0]);

        let src = src_orig;
        let mut dst = [0u8; 13];
        assert_eq!(fill_via_chunks(&src, &mut dst), (3, 12));
        assert_eq!(dst, [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0]);

        let src = src_orig;
        let mut dst = [0u8; 5];
        assert_eq!(fill_via_chunks(&src, &mut dst), (2, 5));
        assert_eq!(dst, [1, 0, 0, 0, 2]);
    }

    #[test]
    fn test_fill_via_u64_chunks() {
        let src_orig = [1u64, 2];

        let src = src_orig;
        let mut dst = [0u8; 11];
        assert_eq!(fill_via_chunks(&src, &mut dst), (2, 11));
        assert_eq!(dst, [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0]);

        let src = src_orig;
        let mut dst = [0u8; 17];
        assert_eq!(fill_via_chunks(&src, &mut dst), (2, 16));
        assert_eq!(dst, [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]);

        let src = src_orig;
        let mut dst = [0u8; 5];
        assert_eq!(fill_via_chunks(&src, &mut dst), (1, 5));
        assert_eq!(dst, [1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_read() {
        let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let mut buf = [0u32; 4];
        read_u32_into(&bytes, &mut buf);
        assert_eq!(buf[0], 0x04030201);
        assert_eq!(buf[3], 0x100F0E0D);

        let mut buf = [0u32; 3];
        read_u32_into(&bytes[1..13], &mut buf); // unaligned
        assert_eq!(buf[0], 0x05040302);
        assert_eq!(buf[2], 0x0D0C0B0A);

        let mut buf = [0u64; 2];
        read_u64_into(&bytes, &mut buf);
        assert_eq!(buf[0], 0x0807060504030201);
        assert_eq!(buf[1], 0x100F0E0D0C0B0A09);

        let mut buf = [0u64; 1];
        read_u64_into(&bytes[7..15], &mut buf); // unaligned
        assert_eq!(buf[0], 0x0F0E0D0C0B0A0908);
    }
}
