/// Sealed trait implemented for `u32` and `u64`.
pub trait Sealed: Default + Copy + TryFrom<usize> {
    type Bytes: Sized + AsRef<[u8]> + for<'a> TryFrom<&'a [u8]>;
    const MAX: Self;

    fn from_usize(val: usize) -> Self;
    fn into_usize(self) -> usize;
    fn to_le_bytes(self) -> Self::Bytes;
    fn from_le_bytes(bytes: Self::Bytes) -> Self;
    fn increment(&mut self);
}

impl Sealed for u32 {
    type Bytes = [u8; 4];
    const MAX: Self = u32::MAX;

    fn from_usize(val: usize) -> Self {
        val.try_into().unwrap()
    }
    fn into_usize(self) -> usize {
        self.try_into().unwrap()
    }
    fn to_le_bytes(self) -> Self::Bytes {
        u32::to_le_bytes(self)
    }
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        u32::from_le_bytes(bytes)
    }
    fn increment(&mut self) {
        *self += 1;
    }
}

impl Sealed for u64 {
    type Bytes = [u8; 8];
    const MAX: Self = u64::MAX;

    fn from_usize(val: usize) -> Self {
        val.try_into().unwrap()
    }
    fn into_usize(self) -> usize {
        self.try_into().unwrap()
    }
    fn to_le_bytes(self) -> Self::Bytes {
        u64::to_le_bytes(self)
    }
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        u64::from_le_bytes(bytes)
    }
    fn increment(&mut self) {
        *self += 1;
    }
}
