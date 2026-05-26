//! Minimal DADA2-rs container types needed by the vendored ends-free NW aligner.

#[derive(Debug, Clone, Default)]
pub struct Comparison {
    pub i: u32,
    pub index: u32,
    pub lambda: f64,
    pub hamming: u32,
}

#[derive(Debug, Clone)]
pub struct Sub {
    pub len0: u32,
    pub map: Vec<u16>,
    pub pos: Vec<u16>,
    pub nt0: Vec<u8>,
    pub nt1: Vec<u8>,
    pub q0: Vec<u8>,
    pub q1: Vec<u8>,
}

impl Sub {
    pub fn nsubs(&self) -> usize { self.pos.len() }
}

pub struct Raw {
    /// DADA2 integer encoding: A=1, C=2, G=3, T=4, N=5.
    pub seq: Vec<u8>,
    pub qual: Option<Vec<u8>>,
    pub prior: bool,
    pub kmer: Option<Vec<u16>>,
    pub kmer8: Option<Vec<u8>>,
    pub kord: Option<Vec<u16>>,
    pub reads: u32,
    pub index: u32,
    pub p: f64,
    pub e_minmax: f64,
    pub comp: Comparison,
    pub lock: bool,
    pub correct: bool,
}

impl Raw {
    pub fn new(seq: Vec<u8>, qual: Option<&[f64]>, reads: u32, prior: bool) -> Self {
        Self {
            seq,
            qual: qual.map(|q| q.iter().map(|&v| v.round() as u8).collect()),
            prior,
            kmer: None,
            kmer8: None,
            kord: None,
            reads,
            index: 0,
            p: 0.0,
            e_minmax: -999.0,
            comp: Comparison::default(),
            lock: false,
            correct: true,
        }
    }
    pub fn len(&self) -> usize { self.seq.len() }
}
