use crate::derep::{dereplicate, ReadRecord};
use crate::distance::corrected_kmer_dist_full;
use crate::error::{ResolvoError, Result};
use crate::kmer::dense_counts;
use crate::sketch::{KmerUtilsOptDensSketcher, SketchDistance};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct FadParams {
    pub k: usize,
    pub neighbor_threshold: f64,
    pub min_count: u64,
    pub use_sketch: bool,
    pub sketch_size: usize,
    pub sketch_prefilter_threshold: f64,
    pub canonical: bool,
}

impl Default for FadParams {
    fn default() -> Self {
        Self {
            k: 6,
            neighbor_threshold: 1.0,
            min_count: 2,
            use_sketch: true,
            sketch_size: 512,
            sketch_prefilter_threshold: 0.03,
            canonical: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FadTemplate {
    pub unique_index: usize,
    pub count: u64,
    pub assigned_count: u64,
}

#[derive(Clone, Debug)]
pub struct FadResult {
    pub unique: Vec<crate::derep::DerepRecord>,
    pub templates: Vec<FadTemplate>,
    pub assignments: Vec<usize>, // per unique index -> template index
}

#[cfg(feature = "parallel")]
fn compute_counts(unique: &[crate::derep::DerepRecord], params: &FadParams) -> Result<Vec<Vec<u16>>> {
    unique
        .par_iter()
        .map(|r| dense_counts(&r.seq, params.k, params.canonical))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn compute_counts(unique: &[crate::derep::DerepRecord], params: &FadParams) -> Result<Vec<Vec<u16>>> {
    unique
        .iter()
        .map(|r| dense_counts(&r.seq, params.k, params.canonical))
        .collect()
}

#[cfg(feature = "parallel")]
fn compute_sketches(unique: &[crate::derep::DerepRecord], params: &FadParams) -> Result<Option<Vec<Vec<u16>>>> {
    if !params.use_sketch { return Ok(None); }
    let sk = KmerUtilsOptDensSketcher::new(params.k, params.sketch_size).canonical(params.canonical);
    unique
        .par_iter()
        .map(|r| sk.sketch(&r.seq).map(|s| s.sig))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

#[cfg(not(feature = "parallel"))]
fn compute_sketches(unique: &[crate::derep::DerepRecord], params: &FadParams) -> Result<Option<Vec<Vec<u16>>>> {
    if !params.use_sketch { return Ok(None); }
    let sk = KmerUtilsOptDensSketcher::new(params.k, params.sketch_size).canonical(params.canonical);
    unique
        .iter()
        .map(|r| sk.sketch(&r.seq).map(|s| s.sig))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// FAD: abundance-ordered template selection with corrected k-mer distances.
///
/// Parallelized stages:
/// - dereplication via parallel fold/reduce
/// - dense k-mer count construction
/// - OptDens sketch construction
/// - nearest-template reassignment
///
/// Template acceptance is intentionally kept serial because FAD is abundance-ordered
/// and later decisions depend on the accepted-template set from earlier uniques.
pub fn fad_denoise(reads: &[ReadRecord], params: &FadParams) -> Result<FadResult> {
    if reads.is_empty() { return Err(ResolvoError::EmptyInput); }
    let unique = dereplicate(reads);
    let n = unique.len();
    let counts = compute_counts(&unique, params)?;
    let sketches = compute_sketches(&unique, params)?;

    let mut templates: Vec<FadTemplate> = Vec::new();
    let mut accepted_unique_indices: Vec<usize> = Vec::new();

    for i in 0..n {
        if unique[i].count < params.min_count { continue; }
        if templates.is_empty() {
            templates.push(FadTemplate { unique_index: i, count: unique[i].count, assigned_count: 0 });
            accepted_unique_indices.push(i);
            continue;
        }
        let mut has_neighbor = false;
        for &ti in &accepted_unique_indices {
            if let Some(ref sigs) = sketches {
                let approx = SketchDistance::Bindash { k: params.k }.eval(&sigs[i], &sigs[ti]);
                if approx > params.sketch_prefilter_threshold { continue; }
            }
            let d = corrected_kmer_dist_full(&counts[i], &counts[ti], params.k);
            if d <= params.neighbor_threshold {
                has_neighbor = true;
                break;
            }
        }
        if !has_neighbor {
            templates.push(FadTemplate { unique_index: i, count: unique[i].count, assigned_count: 0 });
            accepted_unique_indices.push(i);
        }
    }

    if templates.is_empty() {
        templates.push(FadTemplate { unique_index: 0, count: unique[0].count, assigned_count: 0 });
        accepted_unique_indices.push(0);
    }

    let assignments = assign_to_templates(&counts, &accepted_unique_indices, params.k);
    let mut templates = templates;
    for t in &mut templates { t.assigned_count = 0; }
    for (i, &tidx) in assignments.iter().enumerate() { templates[tidx].assigned_count += unique[i].count; }

    Ok(FadResult { unique, templates, assignments })
}

#[cfg(feature = "parallel")]
fn assign_to_templates(counts: &[Vec<u16>], accepted_unique_indices: &[usize], k: usize) -> Vec<usize> {
    counts
        .par_iter()
        .map(|kv| nearest_template(kv, counts, accepted_unique_indices, k))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn assign_to_templates(counts: &[Vec<u16>], accepted_unique_indices: &[usize], k: usize) -> Vec<usize> {
    counts
        .iter()
        .map(|kv| nearest_template(kv, counts, accepted_unique_indices, k))
        .collect()
}

fn nearest_template(kv: &[u16], counts: &[Vec<u16>], accepted_unique_indices: &[usize], k: usize) -> usize {
    let mut best_t = 0usize;
    let mut best_d = f64::INFINITY;
    for (tidx, &ui) in accepted_unique_indices.iter().enumerate() {
        let d = corrected_kmer_dist_full(kv, &counts[ui], k);
        if d < best_d { best_d = d; best_t = tidx; }
    }
    best_t
}
