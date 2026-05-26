use crate::align::{needleman_wunsch, AlignParams};
use crate::dna2bit::Dna2Bit;
use crate::distance::corrected_kmer_dist_full;
use crate::error::Result;
use crate::kmer::dense_counts;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct ConsensusParams {
    pub k: usize,
    pub min_base_fraction: f64,
    pub polish_rounds: usize,
    pub align: AlignParams,
}

impl Default for ConsensusParams {
    fn default() -> Self {
        Self { k: 6, min_base_fraction: 0.5, polish_rounds: 1, align: AlignParams::default() }
    }
}

pub fn choose_centroid(reads: &[Dna2Bit], k: usize) -> Result<usize> {
    if reads.len() <= 1 { return Ok(0); }
    let counts = compute_counts(reads, k)?;
    let sums = pairwise_distance_sums(&counts, k);
    let (best, _) = sums
        .into_iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.0));
    Ok(best)
}

#[cfg(feature = "parallel")]
fn compute_counts(reads: &[Dna2Bit], k: usize) -> Result<Vec<Vec<u16>>> {
    reads.par_iter().map(|r| dense_counts(r, k, true)).collect()
}

#[cfg(not(feature = "parallel"))]
fn compute_counts(reads: &[Dna2Bit], k: usize) -> Result<Vec<Vec<u16>>> {
    reads.iter().map(|r| dense_counts(r, k, true)).collect()
}

#[cfg(feature = "parallel")]
fn pairwise_distance_sums(counts: &[Vec<u16>], k: usize) -> Vec<f64> {
    counts
        .par_iter()
        .enumerate()
        .map(|(i, ci)| {
            counts
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, cj)| corrected_kmer_dist_full(ci, cj, k))
                .sum()
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn pairwise_distance_sums(counts: &[Vec<u16>], k: usize) -> Vec<f64> {
    counts
        .iter()
        .enumerate()
        .map(|(i, ci)| {
            counts
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, cj)| corrected_kmer_dist_full(ci, cj, k))
                .sum()
        })
        .collect()
}

/// Build a simple multiple alignment pileup to a centroid reference and call a majority consensus.
/// Insertions relative to the reference are emitted when the same insertion column receives support.
///
/// Parallelized stages:
/// - centroid k-mer counting and pairwise centroid search
/// - per-read alignments and pileup construction, reduced into one pileup
pub fn consensus_from_cluster(reads: &[Dna2Bit], params: &ConsensusParams) -> Result<Dna2Bit> {
    if reads.is_empty() { return Ok(Dna2Bit::from_codes(&[])); }
    if reads.len() == 1 { return Ok(reads[0].clone()); }
    let mut reference = reads[choose_centroid(reads, params.k)?].clone();
    for _ in 0..params.polish_rounds.max(1) {
        reference = polish_once(&reference, reads, params)?;
    }
    Ok(reference)
}

#[derive(Clone)]
struct Pileup {
    base_counts: Vec<[u32; 4]>,
    del_counts: Vec<u32>,
    ins_counts: Vec<Vec<[u32; 4]>>,
}

impl Pileup {
    fn new(rlen: usize) -> Self {
        Self {
            base_counts: vec![[0u32; 4]; rlen],
            del_counts: vec![0u32; rlen],
            ins_counts: vec![Vec::new(); rlen + 1],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (a, b) in self.base_counts.iter_mut().zip(other.base_counts) {
            for i in 0..4 { a[i] += b[i]; }
        }
        for (a, b) in self.del_counts.iter_mut().zip(other.del_counts) { *a += b; }
        for (slot, other_slot) in self.ins_counts.iter_mut().zip(other.ins_counts) {
            if slot.len() < other_slot.len() { slot.resize(other_slot.len(), [0u32; 4]); }
            for (a, b) in slot.iter_mut().zip(other_slot) {
                for i in 0..4 { a[i] += b[i]; }
            }
        }
        self
    }

    fn add_alignment(&mut self, reference: &Dna2Bit, read: &Dna2Bit, params: &ConsensusParams) {
        let aln = needleman_wunsch(reference, read, &params.align);
        let mut rpos = 0usize;
        let mut ins_pos = 0usize;
        let mut last_ref_pos = 0usize;
        for (ra, qa) in aln.ref_aln.iter().zip(aln.qry_aln.iter()) {
            match (ra, qa) {
                (Some(_rb), Some(qb)) => {
                    self.base_counts[rpos][*qb as usize] += 1;
                    last_ref_pos = rpos + 1;
                    rpos += 1;
                    ins_pos = 0;
                }
                (Some(_), None) => {
                    self.del_counts[rpos] += 1;
                    last_ref_pos = rpos + 1;
                    rpos += 1;
                    ins_pos = 0;
                }
                (None, Some(qb)) => {
                    let slot = &mut self.ins_counts[last_ref_pos];
                    if slot.len() <= ins_pos { slot.push([0u32; 4]); }
                    slot[ins_pos][*qb as usize] += 1;
                    ins_pos += 1;
                }
                (None, None) => {}
            }
        }
    }
}

fn polish_once(reference: &Dna2Bit, reads: &[Dna2Bit], params: &ConsensusParams) -> Result<Dna2Bit> {
    let rlen = reference.len();
    let pileup = build_pileup(reference, reads, params);
    let min_support = (reads.len() as f64 * params.min_base_fraction).ceil() as u32;
    let mut out = Vec::new();
    for pos in 0..=rlen {
        // insertions before reference pos `pos`
        for col in &pileup.ins_counts[pos] {
            let (base, count) = argmax4(col);
            if count >= min_support { out.push(base as u8); }
        }
        if pos == rlen { break; }
        let (base, count) = argmax4(&pileup.base_counts[pos]);
        if count >= pileup.del_counts[pos].max(min_support.saturating_sub(1)) {
            out.push(base as u8);
        }
    }
    Ok(Dna2Bit::from_codes(&out))
}

#[cfg(feature = "parallel")]
fn build_pileup(reference: &Dna2Bit, reads: &[Dna2Bit], params: &ConsensusParams) -> Pileup {
    let rlen = reference.len();
    reads
        .par_iter()
        .map(|read| {
            let mut p = Pileup::new(rlen);
            p.add_alignment(reference, read, params);
            p
        })
        .reduce(|| Pileup::new(rlen), |a, b| a.merge(b))
}

#[cfg(not(feature = "parallel"))]
fn build_pileup(reference: &Dna2Bit, reads: &[Dna2Bit], params: &ConsensusParams) -> Pileup {
    let rlen = reference.len();
    let mut p = Pileup::new(rlen);
    for read in reads { p.add_alignment(reference, read, params); }
    p
}

fn argmax4(v: &[u32; 4]) -> (usize, u32) {
    let mut best = 0usize;
    let mut bc = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > bc { best = i; bc = x; }
    }
    (best, bc)
}
