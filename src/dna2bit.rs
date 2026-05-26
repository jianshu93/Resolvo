use crate::error::{ResolvoError, Result};

/// Compact 2-bit DNA representation. Ambiguous bases are rejected by design.
///
/// Encoding: A=0, C=1, G=2, T=3.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Dna2Bit {
    len: usize,
    words: Vec<u64>,
}

#[inline]
pub fn encode_base(b: u8) -> Option<u8> {
    match b.to_ascii_uppercase() {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'T' | b'U' => Some(3),
        _ => None,
    }
}

#[inline]
pub fn decode_base(x: u8) -> u8 {
    match x & 3 {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        _ => b'T',
    }
}

#[inline]
pub fn comp_base(x: u8) -> u8 { 3 - (x & 3) }

impl Dna2Bit {
    pub fn from_ascii(seq: &[u8]) -> Result<Self> {
        let len = seq.len();
        let mut words = vec![0u64; (len + 31) / 32];
        for (i, &b) in seq.iter().enumerate() {
            let code = encode_base(b).ok_or(ResolvoError::InvalidBase { base: b, pos: i })? as u64;
            let w = i / 32;
            let shift = 2 * (i % 32);
            words[w] |= code << shift;
        }
        Ok(Self { len, words })
    }

    pub fn from_codes(codes: &[u8]) -> Self {
        let len = codes.len();
        let mut words = vec![0u64; (len + 31) / 32];
        for (i, &code) in codes.iter().enumerate() {
            let w = i / 32;
            let shift = 2 * (i % 32);
            words[w] |= ((code & 3) as u64) << shift;
        }
        Self { len, words }
    }

    #[inline]
    pub fn len(&self) -> usize { self.len }
    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline]
    pub fn words(&self) -> &[u64] { &self.words }

    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        debug_assert!(i < self.len);
        let w = i / 32;
        let shift = 2 * (i % 32);
        ((self.words[w] >> shift) & 3) as u8
    }

    pub fn to_ascii(&self) -> Vec<u8> {
        (0..self.len).map(|i| decode_base(self.get(i))).collect()
    }

    /// DADA2-rs-compatible byte encoding: A=1,C=2,G=3,T=4.
    pub fn to_dada_encoded(&self) -> Vec<u8> {
        (0..self.len).map(|i| self.get(i) + 1).collect()
    }

    pub fn reverse_complement(&self) -> Self {
        let mut codes = Vec::with_capacity(self.len);
        for i in (0..self.len).rev() {
            codes.push(comp_base(self.get(i)));
        }
        Self::from_codes(&codes)
    }
}

impl std::fmt::Display for Dna2Bit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = String::from_utf8_lossy(&self.to_ascii()).into_owned();
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let d = Dna2Bit::from_ascii(b"ACGTACGT").unwrap();
        assert_eq!(d.to_ascii(), b"ACGTACGT");
        assert_eq!(d.reverse_complement().to_ascii(), b"ACGTACGT");
    }
}
