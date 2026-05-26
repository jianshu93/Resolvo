use crate::dna2bit::Dna2Bit;
use crate::error::{ResolvoError, Result};

pub const MAX_EXACT_K: usize = 15; // 2*k <= 30 fits comfortably in u64 indices.

#[inline]
pub fn n_kmers(k: usize) -> Result<usize> {
    if !(1..=MAX_EXACT_K).contains(&k) {
        return Err(ResolvoError::InvalidK(k));
    }
    Ok(1usize << (2 * k))
}

#[inline]
pub fn kmer_mask(k: usize) -> u64 { (1u64 << (2 * k)) - 1 }

pub struct KmerIter<'a> {
    seq: &'a Dna2Bit,
    k: usize,
    pos: usize,
    code: u64,
    mask: u64,
}

impl<'a> KmerIter<'a> {
    pub fn new(seq: &'a Dna2Bit, k: usize) -> Result<Self> {
        if !(1..=MAX_EXACT_K).contains(&k) { return Err(ResolvoError::InvalidK(k)); }
        Ok(Self { seq, k, pos: 0, code: 0, mask: kmer_mask(k) })
    }
}

impl<'a> Iterator for KmerIter<'a> {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.seq.len() {
            self.code = ((self.code << 2) | self.seq.get(self.pos) as u64) & self.mask;
            self.pos += 1;
            if self.pos >= self.k { return Some(self.code); }
        }
        None
    }
}

#[inline]
pub fn revcomp_kmer(mut code: u64, k: usize) -> u64 {
    let mut rc = 0u64;
    for _ in 0..k {
        let b = code & 3;
        rc = (rc << 2) | (3 - b);
        code >>= 2;
    }
    rc
}

#[inline]
pub fn canonical_kmer(code: u64, k: usize) -> u64 {
    code.min(revcomp_kmer(code, k))
}

pub fn dense_counts(seq: &Dna2Bit, k: usize, canonical: bool) -> Result<Vec<u16>> {
    if seq.len() < k { return Err(ResolvoError::SequenceTooShort { len: seq.len(), k }); }
    let mut counts = vec![0u16; n_kmers(k)?];
    for code in KmerIter::new(seq, k)? {
        let c = if canonical { canonical_kmer(code, k) } else { code } as usize;
        counts[c] = counts[c].saturating_add(1);
    }
    Ok(counts)
}

pub fn sparse_counts(seq: &Dna2Bit, k: usize, canonical: bool) -> Result<Vec<(u64, u32)>> {
    if seq.len() < k { return Err(ResolvoError::SequenceTooShort { len: seq.len(), k }); }
    let mut v: Vec<u64> = KmerIter::new(seq, k)?.map(|code| if canonical { canonical_kmer(code, k) } else { code }).collect();
    v.sort_unstable();
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let code = v[i];
        let mut j = i + 1;
        while j < v.len() && v[j] == code { j += 1; }
        out.push((code, (j - i) as u32));
        i = j;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_work() {
        let d = Dna2Bit::from_ascii(b"AAAAA").unwrap();
        let c = dense_counts(&d, 2, false).unwrap();
        assert_eq!(c[0], 4);
    }
}
