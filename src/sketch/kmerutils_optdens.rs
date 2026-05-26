//! OptDens sketching through kmerutils, not a reimplementation.
//!
//! This wraps the exact API pattern from the uploaded Bindash code:
//! `kmerutils::sketching::setsketchert::OptDensHashSketch` produces lower-16
//! OptDens signatures (`Vec<u16>`), and sketch comparisons are done with
//! `anndists::dist::DistHamming` in `sketch::ann`.

use crate::dna2bit::Dna2Bit;
use crate::error::Result;
use kmerutils::base::{
    alphabet::Alphabet2b,
    kmergenerator::{KmerGenerationPattern, KmerGenerator, KmerSeqIterator},
    sequence::Sequence as KmerSequence,
    CompressedKmerT,
    KmerT,
    Kmer32bit,
    KmerBuilder,
};
use kmerutils::sketcharg::{DataType, SeqSketcherParams, SketchAlgo};
use kmerutils::sketching::setsketchert::{OptDensHashSketch, SeqSketcherT};

#[derive(Clone, Debug)]
pub struct OptDensSketch {
    pub sig: Vec<u16>,
}

#[derive(Clone, Debug)]
pub struct KmerUtilsOptDensSketcher {
    pub k: usize,
    pub sketch_size: usize,
    pub canonical: bool,
}

impl KmerUtilsOptDensSketcher {
    pub fn new(k: usize, sketch_size: usize) -> Self {
        Self { k, sketch_size, canonical: true }
    }

    pub fn canonical(mut self, canonical: bool) -> Self {
        self.canonical = canonical;
        self
    }

    pub fn sketch(&self, seq: &Dna2Bit) -> Result<OptDensSketch> {
        // kmerutils Sequence is already 2-bit compressed internally.
        let alphabet = Alphabet2b::new();
        let ascii = seq.to_ascii();
        let mut kseq = KmerSequence::with_capacity(2, ascii.len());
        kseq.encode_and_add(&ascii, &alphabet);

        let sketch_args = SeqSketcherParams::new(
            self.k,
            self.sketch_size,
            SketchAlgo::OPTDENS,
            DataType::DNA,
        );
        let sketcher = OptDensHashSketch::<Kmer32bit, f32>::new(&sketch_args);

        let nb_alphabet_bits = 2;
        let canonical = self.canonical;
        let kmer_hash_fn = move |kmer: &Kmer32bit| -> <Kmer32bit as CompressedKmerT>::Val {
            let mask: <Kmer32bit as CompressedKmerT>::Val =
                num::NumCast::from::<u64>((1u64 << (nb_alphabet_bits * kmer.get_nb_base())) - 1)
                    .unwrap();
            let fw = kmer.get_compressed_value() & mask;
            if !canonical {
                return fw;
            }
            // For amplicons that are primer-oriented, canonical=false is often best.
            // With canonical=true, use a cheap reverse-complement on the 2-bit value.
            let rc = revcomp_kmer32(fw as u64, kmer.get_nb_base() as usize) as <Kmer32bit as CompressedKmerT>::Val;
            fw.min(rc)
        };

        let refs = [&kseq];
        let sig = sketcher.sketch_compressedkmer(&refs, kmer_hash_fn);
        Ok(OptDensSketch { sig: sig.into_iter().next().unwrap_or_default() })
    }
}

#[inline]
fn revcomp_kmer32(mut code: u64, k: usize) -> u64 {
    let mut rc = 0u64;
    for _ in 0..k {
        let b = code & 3;
        rc = (rc << 2) | (3 - b);
        code >>= 2;
    }
    rc
}
