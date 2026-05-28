use crate::dada_nwalign::{align_endsfree_with_buf, align_vectorized_with_buf, AlignBuffers, VectorizedAlignScores};
use crate::dna2bit::Dna2Bit;

#[derive(Clone, Debug)]
pub struct Alignment {
    pub ref_aln: Vec<Option<u8>>, // base code 0..3 or gap
    pub qry_aln: Vec<Option<u8>>,
    pub score: i32,
}

#[derive(Clone, Debug)]
pub struct AlignParams {
    pub match_score: i32,
    pub mismatch_score: i32,
    pub gap_score: i32,
    pub band: i32,
    pub vectorized: bool,
}

impl Default for AlignParams {
    fn default() -> Self {
        Self { match_score: 2, mismatch_score: -2, gap_score: -3, band: 16, vectorized: true }
    }
}

/// Fast ends-free NW wrapper around the DADA2-rs aligner.
///
/// Uses a DADA2-style positive band by default (`band = 16`), where a negative
/// band disables banding. For full-length/operon reads, pass a larger band
/// through the CLI, e.g. `--align-band 128`.
///
/// The SIMD/vectorized anti-diagonal path is allowed up to 8000 bp. This is
/// safe for the default ResolvO scoring used here: match = 2, mismatch = -2,
/// gap = -3. A perfect 8000 bp alignment scores 16,000, below i16::MAX.
pub fn endsfree_nw(reference: &Dna2Bit, query: &Dna2Bit, p: &AlignParams) -> Alignment {
    let s1 = reference.to_dada_encoded();
    let s2 = query.to_dada_encoded();
    let mut buf = AlignBuffers::new();
    const VECTORIZED_MAX_LEN: usize = 8000;
    if p.vectorized && s1.len() <= VECTORIZED_MAX_LEN && s2.len() <= VECTORIZED_MAX_LEN {
        align_vectorized_with_buf(&s1, &s2, &VectorizedAlignScores {
            match_score: p.match_score as i16,
            mismatch: p.mismatch_score as i16,
            gap_p: p.gap_score as i16,
            end_gap_p: 0,
            band: p.band,
        }, &mut buf);
    } else {
        align_endsfree_with_buf(&s1, &s2, p.match_score, p.mismatch_score, p.gap_score, p.band, &mut buf);
    }
    let (a0, a1) = buf.alignment();
    let mut ref_aln = Vec::with_capacity(a0.len());
    let mut qry_aln = Vec::with_capacity(a1.len());
    for (&x, &y) in a0.iter().zip(a1.iter()) {
        ref_aln.push(dada_aln_byte_to_code(x));
        qry_aln.push(dada_aln_byte_to_code(y));
    }
    Alignment { ref_aln, qry_aln, score: 0 }
}

/// Backwards-compatible name used by the consensus module.
pub fn needleman_wunsch(reference: &Dna2Bit, query: &Dna2Bit, p: &AlignParams) -> Alignment {
    endsfree_nw(reference, query, p)
}

#[inline]
fn dada_aln_byte_to_code(b: u8) -> Option<u8> {
    match b {
        b'-' => None,
        1..=4 => Some(b - 1),
        _ => None,
    }
}
