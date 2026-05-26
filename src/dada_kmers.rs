//! Compatibility k-mer distance functions for the vendored DADA2-rs aligner.
//!
//! Resolvo's FAD/RAD code uses `crate::kmer` and `crate::distance` directly.
//! These functions only satisfy `dada_nwalign::raw_align_with_buf` when callers
//! enable its optional k-mer screen. Resolvo consensus calls the raw ends-free
//! functions directly and does not use this module on the hot path.

pub fn kmer_dist(k1: &[u16], _len1: usize, k2: &[u16], _len2: usize, k: usize) -> f64 {
    let mut sq = 0u64;
    let mut s1 = 0u64;
    let mut s2 = 0u64;
    for (&a, &b) in k1.iter().zip(k2.iter()) {
        let d = a as i64 - b as i64;
        sq += (d * d) as u64;
        s1 += a as u64;
        s2 += b as u64;
    }
    if s1 + s2 == 0 { 0.0 } else { sq as f64 / (k as f64 * (s1 + s2) as f64) }
}

pub fn kmer_dist8(k1: &[u8], _len1: usize, k2: &[u8], _len2: usize, k: usize) -> f64 {
    let mut sq = 0u64;
    let mut s1 = 0u64;
    let mut s2 = 0u64;
    for (&a, &b) in k1.iter().zip(k2.iter()) {
        let d = a as i64 - b as i64;
        sq += (d * d) as u64;
        s1 += a as u64;
        s2 += b as u64;
    }
    if s1 + s2 == 0 { 0.0 } else { sq as f64 / (k as f64 * (s1 + s2) as f64) }
}

pub fn kord_dist(o1: &[u16], _len1: usize, o2: &[u16], _len2: usize, _k: usize) -> f64 {
    let n = o1.len().min(o2.len());
    if n == 0 { return 1.0; }
    let mismatches = o1.iter().take(n).zip(o2.iter().take(n)).filter(|(a,b)| a != b).count();
    mismatches as f64 / n as f64
}
