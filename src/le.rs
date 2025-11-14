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
//! We provide the [`read_words_into`] helper function for this. The examples below
//! demonstrate how it can be used in practice.
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
//! # Examples
//!
//! The examples below demonstrate how functions in this module can be used to implement
//! [`RngCore`] and [`SeedableRng`] for common RNG algorithm classes.
//!
//! WARNING: the step RNG implementations below are provided for demonstation purposes only and
//! should not be used in practice!
//!
//! ## Single 32-bit value RNG
//!
//! ```
//! use rand_core::{RngCore, SeedableRng, le};
//!
//! pub struct Step32Rng(u32);
//!
//! impl SeedableRng for Step32Rng {
//!     type Seed = [u8; 4];
//!
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         // Always use Little-Endian conversion to ensure portable results
//!         Self(u32::from_le_bytes(seed))
//!     }
//! }
//!
//! impl RngCore for Step32Rng {
//!     fn next_u32(&mut self) -> u32 {
//!         let val = self.0;
//!         self.0 = val + 1;
//!         val
//!     }
//!
//!     fn next_u64(&mut self) -> u64 {
//!         le::next_u64_via_u32(self)
//!     }
//!
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         le::fill_bytes_via_next_word(dst, || self.next_u32());
//!     }
//! }
//!
//! let mut rng = Step32Rng::seed_from_u64(42);
//!
//! assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! assert_eq!(rng.next_u64(), 0x7ba1_8fa6_7ba1_8fa5);
//! let mut buf = [0u8; 5];
//! rng.fill_bytes(&mut buf);
//! assert_eq!(buf, [0xa7, 0x8f, 0xa1, 0x7b, 0xa8]);
//! ```
//!
//! ## Single 64-bit value RNG
//!
//! ```
//! use rand_core::{RngCore, SeedableRng, le};
//!
//! pub struct Step64Rng(u64);
//!
//! impl SeedableRng for Step64Rng {
//!     type Seed = [u8; 8];
//!
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         Self(u64::from_le_bytes(seed))
//!     }
//! }
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
//!         le::fill_bytes_via_next_word(dst, || self.next_u64());
//!     }
//! }
//!
//! let mut rng = Step64Rng::seed_from_u64(42);
//!
//! assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! assert_eq!(rng.next_u64(), 0x0a3d_3258_7ba1_8fa5);
//! let mut buf = [0u8; 5];
//! rng.fill_bytes(&mut buf);
//! assert_eq!(buf, [0xa6, 0x8f, 0xa1, 0x7b, 0x58]);
//! ```
//!
//! ## 32-bit block RNG
//!
//! ```
//! use rand_core::{RngCore, SeedableRng, le};
//!
//! struct Step8x32RngCore([u32; 8]);
//!
//! impl Step8x32RngCore {
//!     fn next_block(&mut self, block: &mut [u32; 8]) {
//!         *block = self.0;
//!         self.0.iter_mut().for_each(|v| *v += 1);
//!     }
//! }
//!
//! pub struct Step8x32Rng {
//!     core: Step8x32RngCore,
//!     buffer: [u32; 8],
//! }
//!
//! impl SeedableRng for Step8x32Rng {
//!     type Seed = [u8; 32];
//!
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         let mut core_state = [0u32; 8];
//!         le::read_words_into(&seed, &mut core_state);
//!         Self {
//!             core: Step8x32RngCore(core_state),
//!             buffer: le::new_buffer(),
//!         }
//!     }
//! }
//!
//! impl RngCore for Step8x32Rng {
//!     fn next_u32(&mut self) -> u32 {
//!         le::next_word_via_gen_block(&mut self.buffer, |block| self.core.next_block(block))
//!     }
//!
//!     fn next_u64(&mut self) -> u64 {
//!         le::next_u64_via_u32(self)
//!     }
//!
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         le::fill_bytes_via_next_word(dst, || self.next_u32());
//!     }
//! }
//!
//! let mut rng = Step8x32Rng::seed_from_u64(42);
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
//! use rand_core::{RngCore, SeedableRng, le};
//!
//! struct Block64RngCore {
//!     // ...
//!     # state: [u64; 4],
//! }
//!
//! impl Block64RngCore {
//!     fn new(seed: [u64; 4]) -> Self {
//!         // ...
//!         # Self { state: seed }
//!     }
//!
//!     fn next_block(&mut self, block: &mut [u64; 4]) {
//!         // ...
//!         # *block = self.state;
//!         # self.state.iter_mut().for_each(|v| *v += 1);
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
//!         let mut seed_u64 = [0u64; 4];
//!         le::read_words_into(&seed, &mut seed_u64);
//!         Self {
//!             core: Block64RngCore::new(seed_u64),
//!             buffer: le::new_buffer(),
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
//!         le::next_word_via_gen_block(&mut self.buffer, |block| self.core.next_block(block))
//!     }
//!
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         le::fill_bytes_via_next_word(dst, || self.next_u64());
//!     }
//! }
//!
//! # let mut rng = Block64Rng::seed_from_u64(42);
//! # assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! # assert_eq!(rng.next_u64(), 0xb814_0169_cca1_b8ea);
//! # let mut buf = [0u8; 5];
//! # rng.fill_bytes(&mut buf);
//! # assert_eq!(buf, [0x2b, 0x8c, 0xc8, 0x75, 0x18]);
//! ```
//!
//! ## Fill-based RNG
//!
//! ```
//! use rand_core::RngCore;
//!
//! pub struct FillRng {
//!     // ...
//!     # state: u8,
//! }
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

use crate::RngCore;

/// Implement `next_u64` via `next_u32` using little-endian order.
#[inline]
pub fn next_u64_via_u32<R: RngCore + ?Sized>(rng: &mut R) -> u64 {
    // Use LE; we explicitly generate one value before the next.
    let x = u64::from(rng.next_u32());
    let y = u64::from(rng.next_u32());
    (y << 32) | x
}

/// Implement `fill_bytes` via `next_u64` using little-endian order.
#[inline]
pub fn fill_bytes_via_next_word<W: Word>(dest: &mut [u8], mut next_word: impl FnMut() -> W) {
    let mut chunks = dest.chunks_exact_mut(size_of::<W>());
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

/// Fills slice of words `dst` from byte slice `src` using little endian order.
///
/// # Panics
///
/// If `size_of_val(src) != size_of_val(dst)`.
#[inline]
pub fn read_words_into<W: Word>(src: &[u8], dst: &mut [W]) {
    assert_eq!(size_of_val(src), size_of_val(dst));
    let chunks = src.chunks_exact(size_of::<W>());
    for (out, chunk) in dst.iter_mut().zip(chunks) {
        let Ok(bytes) = chunk.try_into() else {
            unreachable!()
        };
        *out = W::from_le_bytes(bytes);
    }
}

/// Create new block buffer.
///
/// # Panics
/// If `N` is smaller than 2 or can not be represented as `W`.
#[inline]
pub fn new_buffer<W: Word, const N: usize>() -> [W; N] {
    assert!(N > 2);
    // Check that `N` can be converted into `W`.
    let _ = W::from_usize(N);
    let mut res = [W::from_usize(0); N];
    res[0] = W::from_usize(N);
    res
}

/// Implement `next_u32/u64` function using buffer and block generation closure.
#[inline]
pub fn next_word_via_gen_block<W: Word, const N: usize>(
    buf: &mut [W; N],
    mut generate_block: impl FnMut(&mut [W; N]),
) -> W {
    let pos = buf[0].into_usize();
    debug_assert_ne!(pos, 0, "cursor position should not be zero");
    match buf.get(pos) {
        Some(&val) => {
            buf[0].increment();
            val
        }
        None => {
            generate_block(buf);
            core::mem::replace(&mut buf[0], W::from_usize(1))
        }
    }
}

/// Implement `fill_bytes` using buffer and block generation closure.
#[inline]
pub fn fill_bytes_via_gen_block<W: Word, const N: usize>(
    mut dst: &mut [u8],
    buf: &mut [W; N],
    mut generate_block: impl FnMut(&mut [W; N]),
) {
    let word_size = size_of::<W>();

    let pos = buf[0];
    let pos_usize = pos.into_usize();
    debug_assert_ne!(pos_usize, 0, "cursor position should not be zero");
    if pos_usize < buf.len() {
        let buf_tail = &buf[pos_usize..];
        let buf_rem = size_of_val(buf_tail);

        if buf_rem >= dst.len() {
            let new_pos = read_bytes(buf, dst, pos);
            buf[0] = new_pos;
            return;
        }

        let (l, r) = dst.split_at_mut(buf_rem);
        read_bytes(buf, l, pos);
        dst = r;
    }

    let mut blocks = dst.chunks_exact_mut(N * word_size);
    let zero = W::from_usize(0);
    for block in &mut blocks {
        // We intentionally use the temporary buffer to prevent unnecessary writes
        // to the original `buf` and to enable potential optimization of writing
        // generated data directly into `block`.
        let mut buf = [zero; N];
        generate_block(&mut buf);
        read_bytes(&buf, block, zero);
    }

    let rem = blocks.into_remainder();
    let new_pos = if rem.is_empty() {
        W::from_usize(N)
    } else {
        generate_block(buf);
        read_bytes::<W, N>(buf, rem, zero)
    };
    buf[0] = new_pos;
}

/// Reads bytes from `&block[pos..new_pos]` to `dst` using little endian byte order
/// ignoring the tail bytes if necessary and returns `new_pos`.
///
/// This function is written in a way which helps the compiler to compile it down
/// to one `memcpy`. The temporary buffer gets eliminated by the compiler, see:
/// https://rust.godbolt.org/z/T8f77KjGc
#[inline]
fn read_bytes<W: Word, const N: usize>(block: &[W; N], dst: &mut [u8], pos: W) -> W {
    let word_size = size_of::<W>();
    let pos = pos.into_usize();
    assert!(size_of_val(&block[pos..]) >= size_of_val(dst));

    // TODO: replace with `[0u8; { size_of::<W>() * N }]` on
    // stabilization of `generic_const_exprs`
    let mut buf = [W::from_usize(0); N];
    // SAFETY: it's safe to reference `[u32/u64; N]` as `&mut [u8]`
    // with length equal to `size_of::<u32/u64>() * N`
    let buf: &mut [u8] = unsafe {
        let p: *mut u8 = buf.as_mut_ptr().cast();
        let len = word_size * N;
        core::slice::from_raw_parts_mut(p, len)
    };

    for (src, dst) in block.iter().zip(buf.chunks_exact_mut(word_size)) {
        let val = src.to_le_bytes();
        dst.copy_from_slice(val.as_ref())
    }

    let offset = pos * word_size;
    dst.copy_from_slice(&buf[offset..][..dst.len()]);
    let read_words = dst.len().div_ceil(word_size);
    W::from_usize(pos + read_words)
}

/// Sealed trait implemented for `u32` and `u64`.
pub trait Word: crate::sealed::Sealed {}

impl Word for u32 {}
impl Word for u64 {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_read() {
        let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let mut buf = [0u32; 4];
        read_words_into(&bytes, &mut buf);
        assert_eq!(buf[0], 0x04030201);
        assert_eq!(buf[3], 0x100F0E0D);

        let mut buf = [0u32; 3];
        read_words_into(&bytes[1..13], &mut buf); // unaligned
        assert_eq!(buf[0], 0x05040302);
        assert_eq!(buf[2], 0x0D0C0B0A);

        let mut buf = [0u64; 2];
        read_words_into(&bytes, &mut buf);
        assert_eq!(buf[0], 0x0807060504030201);
        assert_eq!(buf[1], 0x100F0E0D0C0B0A09);

        let mut buf = [0u64; 1];
        read_words_into(&bytes[7..15], &mut buf); // unaligned
        assert_eq!(buf[0], 0x0F0E0D0C0B0A0908);
    }
}
