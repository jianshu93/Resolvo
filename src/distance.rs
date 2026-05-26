/// FAD absolute corrected k-mer distance from the RAD/FAD Julia code:
/// sqeuclidean(kmer_counts_a, kmer_counts_b) / (2*k).
pub fn corrected_kmer_dist_full(a: &[u16], b: &[u16], k: usize) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let sq: u64 = a.iter().zip(b).map(|(&x, &y)| {
        let d = x as i64 - y as i64;
        (d * d) as u64
    }).sum();
    sq as f64 / (2.0 * k as f64)
}

/// RAD normalized corrected k-mer distance:
/// sqeuclidean(a,b) / (k * (sum(a) + sum(b))).
pub fn corrected_kmer_dist(a: &[u16], b: &[u16], k: usize) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sq = 0u64;
    let mut sa = 0u64;
    let mut sb = 0u64;
    for (&x, &y) in a.iter().zip(b) {
        let d = x as i64 - y as i64;
        sq += (d * d) as u64;
        sa += x as u64;
        sb += y as u64;
    }
    if sa + sb == 0 { 0.0 } else { sq as f64 / (k as f64 * (sa + sb) as f64) }
}

/// Bindash/kmerutils-style normalized Hamming over lower-16 OptDens signatures.
/// This uses `anndists::dist::DistHamming`, as in the uploaded Bindash code.
pub fn anndists_hamming_u16(a: &[u16], b: &[u16]) -> f64 {
    use anndists::dist::{Distance, DistHamming};
    let dist = DistHamming;
    dist.eval(a, b) as f64
}

/// Bindash-style distance transform from MinHash/OPH Hamming distance.
/// h = normalized sketch Hamming; j = estimated Jaccard = 1 - h.
/// distance = -ln(2j/(1+j)) / k.
pub fn bindash_distance_from_hamming(h: f64, k: usize) -> f64 {
    let h = h.clamp(0.0, 1.0);
    let mut j = 1.0 - h;
    if j <= 0.0 { j = f64::MIN_POSITIVE; }
    let fraction = (2.0 * j) / (1.0 + j);
    -fraction.ln() / k as f64
}
