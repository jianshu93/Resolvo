//! Resolvo: long-amplicon template recovery from noisy long-read clusters.

pub mod dna2bit;
pub mod error;
pub mod io;
pub mod derep;
pub mod kmer;
pub mod logging;
pub mod parallel;
pub mod distance;
pub mod sketch;
pub mod dp_means;
pub mod containers;
pub mod dada_kmers;
pub mod dada_nwalign;
pub mod align;
pub mod consensus;
pub mod algorithms;

pub use dna2bit::Dna2Bit;
pub use derep::{DerepRecord, dereplicate};
pub use algorithms::fad::{FadParams, FadResult, fad_denoise};
pub use algorithms::rad::{RadClusterMode, RadParams, RadResult, rad_denoise};
