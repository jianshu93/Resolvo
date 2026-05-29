use crate::algorithms::fad::{fad_denoise, FadParams};
use crate::consensus::{consensus_from_cluster, ConsensusParams};
use crate::derep::ReadRecord;
use crate::distance::corrected_kmer_dist_full;
use crate::dp_means::{dp_means, normalized_sqeuclidean_centroid, DpMeansParams};
use crate::error::{ResolvoError, Result};
use crate::kmer::dense_counts;
use crate::logging::{cluster_summary, log, StageTimer};
use crate::sketch::{KmerUtilsOptDensSketcher, SketchDistance};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadClusterMode {
    /// Faithful/reference RAD: one global exact dense-kmer DP-means run.
    Exact,
    /// Fast default for large datasets: loose sketch bins first, then exact dense-kmer DP-means inside each bin.
    SketchPreclusterThenExact,
    /// Fastest/most approximate: sketch medoid DP-means only, followed by consensus.
    SketchOnly,
}

impl Default for RadClusterMode {
    fn default() -> Self { Self::SketchPreclusterThenExact }
}

impl std::str::FromStr for RadClusterMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "exact" => Ok(Self::Exact),
            "sketch-precluster" | "sketch_precluster" | "precluster" | "sketch-precluster-then-exact" => {
                Ok(Self::SketchPreclusterThenExact)
            }
            "sketch-only" | "sketch_only" => Ok(Self::SketchOnly),
            other => Err(format!("unknown RAD cluster mode '{other}'")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RadParams {
    pub k: usize,
    pub rough_radius: f64,
    /// Minimum total abundance/support required to keep a cluster.
    /// For dereplicated FASTA, this uses `size=` counts, not unique-record count.
    /// Set to 1 to preserve singleton unique reads through RAD.
    pub min_cluster_size: usize,
    pub consensus: ConsensusParams,
    pub run_fad_clean: bool,
    pub fad: FadParams,
    pub canonical: bool,

    /// Enable Julia-RAD-style recursive fine splitting of rough clusters using high-variance k-mers.
    pub fine_split: bool,
    /// Maximum recursive fine-splitting depth per rough cluster.
    pub fine_split_max_depth: usize,
    /// Number of high-variance k-mer dimensions retained for each split attempt.
    pub fine_split_top_kmers: usize,
    /// Minimum variance required for a k-mer dimension to be considered informative.
    pub fine_split_min_var: f64,
    /// DP-means radius used on the high-variance projected k-mer subspace.
    pub fine_split_radius: f64,
    /// Minimum number of informative k-mers required before attempting a split.
    pub fine_split_min_features: usize,
    /// Exclude trivial homopolymer k-mers from fine-splitting features.
    pub fine_split_drop_homopolymer_kmers: bool,

    /// Select exact RAD, sketch precluster + exact RAD, or sketch-only rough clustering.
    pub cluster_mode: RadClusterMode,
    /// Enable sketches for RAD stages that support candidate shortlisting.
    pub use_sketch: bool,
    /// OptDens signature size.
    pub sketch_size: usize,
    /// Loose Bindash-distance cutoff for sketch candidate prefilters.
    pub sketch_prefilter_threshold: f64,
    /// Maximum approximate candidates to exact-check during final reassignment.
    pub sketch_top_k: usize,
    /// If sketch candidates are empty, fall back to a full exact scan.
    pub sketch_fallback_exact: bool,
    /// Radius for sketch DP-means / sketch rough preclustering.
    pub sketch_radius: f64,
    /// Maximum iterations for sketch-only/precluster medoid assignment.
    pub sketch_max_iter: usize,
    /// Hard cap on sketch-created bins. Prevents pathological one-read-per-bin explosions.
    pub sketch_max_bins: usize,
    /// Maximum iterations for exact dense DP-means.
    pub rough_max_iter: usize,
    /// Maximum iterations for fine-split projected DP-means.
    pub fine_split_max_iter: usize,
    /// Print step-level timing and cluster summaries.
    pub verbose: bool,
}

impl Default for RadParams {
    fn default() -> Self {
        let k = 6;
        Self {
            k,
            rough_radius: 0.01,
            min_cluster_size: 1,
            consensus: ConsensusParams { k, ..ConsensusParams::default() },
            run_fad_clean: true,
            fad: FadParams { k, min_count: 1, neighbor_threshold: 1.0, ..FadParams::default() },
            canonical: true,
            fine_split: true,
            fine_split_max_depth: 4,
            fine_split_top_kmers: 128,
            fine_split_min_var: 0.05,
            fine_split_radius: 0.01,
            fine_split_min_features: 8,
            fine_split_drop_homopolymer_kmers: true,
            cluster_mode: RadClusterMode::SketchPreclusterThenExact,
            use_sketch: true,
            sketch_size: 512,
            sketch_prefilter_threshold: 0.03,
            sketch_top_k: 32,
            sketch_fallback_exact: true,
            sketch_radius: 0.03,
            sketch_max_iter: 8,
            sketch_max_bins: 4096,
            rough_max_iter: 10,
            fine_split_max_iter: 10,
            verbose: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RadTemplate {
    pub seq: crate::Dna2Bit,
    pub cluster_size: usize,
    pub assigned_count: usize,
}

#[derive(Clone, Debug)]
pub struct RadResult {
    pub templates: Vec<RadTemplate>,
    pub read_assignments: Vec<usize>,
    pub clusters: Vec<Vec<usize>>,
}


fn total_input_abundance(reads: &[ReadRecord]) -> u64 {
    reads.iter().map(|r| r.count.max(1)).sum()
}

fn cluster_abundance(reads: &[ReadRecord], members: &[usize]) -> u64 {
    members.iter().map(|&i| reads[i].count.max(1)).sum()
}

fn retain_clusters_by_abundance(reads: &[ReadRecord], clusters: &mut Vec<Vec<usize>>, min_abundance: u64) {
    clusters.retain(|m| cluster_abundance(reads, m) >= min_abundance);
}

fn abundance_cluster_summary(reads: &[ReadRecord], clusters: &[Vec<usize>]) -> String {
    if clusters.is_empty() {
        return "clusters=0 records=0 abundance=0".to_string();
    }
    let mut sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();
    let mut abund: Vec<u64> = clusters.iter().map(|c| cluster_abundance(reads, c)).collect();
    sizes.sort_unstable();
    abund.sort_unstable();
    let records: usize = sizes.iter().sum();
    let abundance: u64 = abund.iter().sum();
    let q = |v: &[usize], p: f64| -> usize { v[((v.len() - 1) as f64 * p).round() as usize] };
    let qa = |v: &[u64], p: f64| -> u64 { v[((v.len() - 1) as f64 * p).round() as usize] };
    format!(
        "clusters={} records={} abundance={} record_size[min={},p50={},p90={},p99={},max={}] abundance[min={},p50={},p90={},p99={},max={}]",
        clusters.len(), records, abundance,
        sizes[0], q(&sizes, 0.50), q(&sizes, 0.90), q(&sizes, 0.99), sizes[sizes.len()-1],
        abund[0], qa(&abund, 0.50), qa(&abund, 0.90), qa(&abund, 0.99), abund[abund.len()-1],
    )
}

/// RAD: k-mer-space clustering + consensus per cluster + optional FAD-clean.
///
/// Clustering modes:
/// - `Exact`: faithful dense-kmer DP-means over all reads.
/// - `SketchPreclusterThenExact`: loose OptDens/Bindash bins, then exact dense-kmer DP-means inside each bin.
/// - `SketchOnly`: OptDens/Bindash medoid DP-means only. Fast but approximate.
///
/// MinHash/OptDens is used only as an acceleration or rough preclustering layer.
/// Exact corrected k-mer distance still makes final reassignment decisions when possible.
pub fn rad_denoise(reads: &[ReadRecord], params: &RadParams) -> Result<RadResult> {
    if reads.is_empty() { return Err(ResolvoError::EmptyInput); }

    log(params.verbose, format!(
        "RAD input records={} total_abundance={} k={} mode={:?} fine_split={} use_sketch={} sketch_size={} align_band={} max_consensus_reads={} rough_max_iter={}",
        reads.len(),
        total_input_abundance(reads),
        params.k,
        params.cluster_mode,
        params.fine_split,
        params.use_sketch,
        params.sketch_size,
        params.consensus.align.band,
        params.consensus.max_consensus_reads,
        params.rough_max_iter,
    ));

    let t = StageTimer::start("compute dense k-mer counts", params.verbose);
    let kmer_vecs = compute_read_counts(reads, params)?;
    t.done();

    let t = StageTimer::start("compute OptDens sketches", params.verbose);
    let read_sketches = compute_read_sketches(reads, params)?;
    t.done();

    let t = StageTimer::start("RAD rough clustering", params.verbose);
    let mut clusters = build_clusters(reads, &kmer_vecs, read_sketches.as_deref(), params)?;
    t.done();
    log(params.verbose, format!("rough {}", abundance_cluster_summary(reads, &clusters)));

    if clusters.is_empty() { clusters.push((0..reads.len()).collect()); }
    retain_clusters_by_abundance(reads, &mut clusters, params.min_cluster_size as u64);
    if clusters.is_empty() { clusters.push((0..reads.len()).collect()); }
    log(params.verbose, format!("after min_cluster_size/support filter {}", abundance_cluster_summary(reads, &clusters)));

    if params.fine_split {
        let t = StageTimer::start("RAD recursive high-variance fine splitting", params.verbose);
        clusters = fine_split_clusters(&kmer_vecs, clusters, params)?;
        t.done();
        retain_clusters_by_abundance(reads, &mut clusters, params.min_cluster_size as u64);
        if clusters.is_empty() { clusters.push((0..reads.len()).collect()); }
        log(params.verbose, format!("fine-split {}", abundance_cluster_summary(reads, &clusters)));
    }

    let t = StageTimer::start("consensus templates", params.verbose);
    let mut templates = consensus_templates(reads, &clusters, params)?;
    t.done();
    log(params.verbose, format!("consensus templates={}", templates.len()));

    if params.run_fad_clean && templates.len() > 1 {
        let t = StageTimer::start("FAD-clean RAD consensus templates", params.verbose);
        let pseudo: Vec<ReadRecord> = templates
            .iter()
            .enumerate()
            .map(|(i, t)| ReadRecord { id: format!("rad_consensus_{i}"), seq: t.seq.clone(), qual: None, count: t.cluster_size as u64 })
            .collect();

        let mut fad_params = params.fad.clone();
        fad_params.k = params.k;
        fad_params.canonical = params.canonical;
        fad_params.use_sketch = params.use_sketch;
        fad_params.sketch_size = params.sketch_size;
        fad_params.sketch_prefilter_threshold = params.sketch_prefilter_threshold;

        let clean = fad_denoise(&pseudo, &fad_params)?;
        let mut kept = Vec::new();
        for tmpl in &clean.templates {
            let uidx = tmpl.unique_index;
            kept.push(RadTemplate {
                seq: clean.unique[uidx].seq.clone(),
                cluster_size: tmpl.assigned_count as usize,
                assigned_count: 0,
            });
        }
        templates = kept;
        t.done();
        log(params.verbose, format!("after FAD-clean templates={}", templates.len()));
    }

    let t = StageTimer::start("template k-mer counts", params.verbose);
    let tmpl_counts = compute_template_counts(&templates, params)?;
    t.done();
    let t = StageTimer::start("template sketches", params.verbose);
    let tmpl_sketches = compute_template_sketches(&templates, params)?;
    t.done();
    let t = StageTimer::start("final read-to-template reassignment", params.verbose);
    let read_assignments = assign_reads_to_templates(
        &kmer_vecs,
        read_sketches.as_deref(),
        &tmpl_counts,
        tmpl_sketches.as_deref(),
        params,
    );
    t.done();
    for t in &mut templates { t.assigned_count = 0; }
    for (i, &a) in read_assignments.iter().enumerate() { templates[a].assigned_count += reads[i].count.max(1) as usize; }

    log(params.verbose, format!("RAD done templates={}", templates.len()));
    Ok(RadResult { templates, read_assignments, clusters })
}

fn build_clusters(
    reads: &[ReadRecord],
    kmer_vecs: &[Vec<u16>],
    sketches: Option<&[Vec<u16>]>,
    params: &RadParams,
) -> Result<Vec<Vec<usize>>> {
    match params.cluster_mode {
        RadClusterMode::Exact => exact_dp_clusters(kmer_vecs, params),
        RadClusterMode::SketchPreclusterThenExact => {
            if !params.use_sketch {
                return exact_dp_clusters(kmer_vecs, params);
            }
            let sketches = sketches.ok_or_else(|| ResolvoError::InvalidParam("sketch precluster requested but sketches are unavailable".into()))?;
            let t = StageTimer::start("sketch LSH pre-binning", params.verbose);
            let bins = sketch_lsh_bins(sketches, params);
            t.done();
            log(params.verbose, format!("sketch bins {}", cluster_summary(&bins)));
            let t = StageTimer::start("exact dense DP-means inside sketch bins", params.verbose);
            let out = exact_dp_within_bins(kmer_vecs, bins, params);
            t.done();
            out
        }
        RadClusterMode::SketchOnly => {
            if !params.use_sketch {
                return exact_dp_clusters(kmer_vecs, params);
            }
            let sketches = sketches.ok_or_else(|| ResolvoError::InvalidParam("sketch-only clustering requested but sketches are unavailable".into()))?;
            let _ = reads; // keep signature symmetric; future versions may use read lengths/ids here.
            Ok(sketch_lsh_bins(sketches, params))
        }
    }
}

fn exact_dp_clusters(kmer_vecs: &[Vec<u16>], params: &RadParams) -> Result<Vec<Vec<usize>>> {
    let dp_params = DpMeansParams { radius: params.rough_radius, max_iter: params.rough_max_iter.max(1) };
    let clusters0 = dp_means(kmer_vecs, &dp_params, |c, v| normalized_sqeuclidean_centroid(c, v, params.k));
    Ok(clusters0.into_iter().map(|c| c.members).collect())
}

#[cfg(feature = "parallel")]
fn exact_dp_within_bins(kmer_vecs: &[Vec<u16>], bins: Vec<Vec<usize>>, params: &RadParams) -> Result<Vec<Vec<usize>>> {
    let nested: Vec<Vec<Vec<usize>>> = bins
        .into_par_iter()
        .map(|bin| exact_dp_one_bin(kmer_vecs, bin, params))
        .collect::<Result<Vec<_>>>()?;
    Ok(nested.into_iter().flatten().collect())
}

#[cfg(not(feature = "parallel"))]
fn exact_dp_within_bins(kmer_vecs: &[Vec<u16>], bins: Vec<Vec<usize>>, params: &RadParams) -> Result<Vec<Vec<usize>>> {
    let mut out = Vec::new();
    for bin in bins {
        out.extend(exact_dp_one_bin(kmer_vecs, bin, params)?);
    }
    Ok(out)
}

fn exact_dp_one_bin(kmer_vecs: &[Vec<u16>], bin: Vec<usize>, params: &RadParams) -> Result<Vec<Vec<usize>>> {
    if bin.len() <= params.min_cluster_size.max(1) {
        return Ok(vec![bin]);
    }
    let local: Vec<Vec<u16>> = bin.iter().map(|&i| kmer_vecs[i].clone()).collect();
    let dp_params = DpMeansParams { radius: params.rough_radius, max_iter: params.rough_max_iter.max(1) };
    let local_clusters = dp_means(&local, &dp_params, |c, v| normalized_sqeuclidean_centroid(c, v, params.k));
    let mapped = local_clusters
        .into_iter()
        .map(|c| c.members.into_iter().map(|local_i| bin[local_i]).collect::<Vec<_>>())
        .collect();
    Ok(mapped)
}


/// Fast sketch pre-binning using OptDens/Bindash signatures.
///
/// The previous version used medoid DP-means over sketches. That still required
/// comparing every read to a growing set of medoids and can be very slow for
/// 100k+ full-operon reads. This routine is intentionally linear-ish:
///
/// - hash several small bands of each 16-bit OptDens signature;
/// - assign each read to the least-occupied deterministic band bucket;
/// - split any oversized bucket deterministically by additional sketch positions.
///
/// These bins are only rough preclusters. In `SketchPreclusterThenExact` mode,
/// exact dense-kmer DP-means still runs inside each bin, and final reassignment
/// is global. Therefore this stage should be fast and high-throughput rather than
/// an expensive final clustering decision.
fn sketch_lsh_bins(sketches: &[Vec<u16>], params: &RadParams) -> Vec<Vec<usize>> {
    if sketches.is_empty() { return Vec::new(); }
    let n = sketches.len();
    let sig_len = sketches[0].len().max(1);
    let band_width = 4usize.min(sig_len).max(1);
    let n_bands = (sig_len / band_width).clamp(1, 16);
    let max_bin_size = rough_max_bin_size(n, params);

    let mut occupancy: HashMap<u64, usize> = HashMap::with_capacity(n.saturating_mul(2).min(1_000_000));
    let mut chosen_keys = Vec::with_capacity(n);

    for (i, sig) in sketches.iter().enumerate() {
        let mut best_key = 0u64;
        let mut best_occ = usize::MAX;
        for b in 0..n_bands {
            let key = sketch_band_key(sig, b, band_width, i % 17);
            let occ = *occupancy.get(&key).unwrap_or(&0);
            if occ < best_occ || (occ == best_occ && key < best_key) {
                best_occ = occ;
                best_key = key;
            }
        }
        *occupancy.entry(best_key).or_insert(0) += 1;
        chosen_keys.push(best_key);
    }

    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::with_capacity(occupancy.len());
    for (i, key) in chosen_keys.into_iter().enumerate() {
        buckets.entry(key).or_default().push(i);
    }

    let mut bins = Vec::new();
    for (_key, bucket) in buckets {
        split_oversized_sketch_bucket(sketches, bucket, max_bin_size, &mut bins);
    }

    bins.retain(|b| !b.is_empty());
    bins
}

fn rough_max_bin_size(n: usize, params: &RadParams) -> usize {
    // Keep exact dense DP-means inside bins bounded. Larger bins can still be
    // expensive because exact DP-means is O(reads × centroids × 4^k).
    // Use sketch_max_bins as a user-facing control over rough granularity: more
    // bins -> smaller target bin size.
    let target_from_bins = (n / params.sketch_max_bins.max(1)).max(params.min_cluster_size.max(16));
    target_from_bins.clamp(256, 2048)
}

fn split_oversized_sketch_bucket(
    sketches: &[Vec<u16>],
    bucket: Vec<usize>,
    max_bin_size: usize,
    out: &mut Vec<Vec<usize>>,
) {
    if bucket.len() <= max_bin_size {
        out.push(bucket);
        return;
    }

    let sig_len = sketches[0].len().max(1);
    let mut sub: HashMap<u64, Vec<usize>> = HashMap::new();
    for &idx in &bucket {
        let sig = &sketches[idx];
        let key = sketch_secondary_key(sig, idx, sig_len);
        sub.entry(key).or_default().push(idx);
    }

    for (_k, mut b) in sub {
        if b.len() <= max_bin_size {
            out.push(b);
        } else {
            // Last-resort deterministic chunking. This is only a pre-binning
            // accelerator; global FAD-clean and final reassignment can still
            // collapse/reassign templates after consensus.
            b.sort_unstable();
            for chunk in b.chunks(max_bin_size) {
                out.push(chunk.to_vec());
            }
        }
    }
}

fn sketch_band_key(sig: &[u16], band: usize, width: usize, salt: usize) -> u64 {
    let mut h = DefaultHasher::new();
    band.hash(&mut h);
    salt.hash(&mut h);
    let start = (band * width) % sig.len().max(1);
    for j in 0..width {
        sig[(start + j) % sig.len()].hash(&mut h);
    }
    h.finish()
}

fn sketch_secondary_key(sig: &[u16], idx: usize, sig_len: usize) -> u64 {
    let mut h = DefaultHasher::new();
    0x9e3779b97f4a7c15u64.hash(&mut h);
    // Spread probes across the signature; use only a few positions to keep this cheap.
    for m in [0usize, 7, 17, 31, 53, 97, 193, 389] {
        sig[(idx.wrapping_mul(131).wrapping_add(m)) % sig_len].hash(&mut h);
    }
    h.finish()
}


#[cfg(feature = "parallel")]
fn fine_split_clusters(kmer_vecs: &[Vec<u16>], clusters: Vec<Vec<usize>>, params: &RadParams) -> Result<Vec<Vec<usize>>> {
    let nested: Vec<Vec<Vec<usize>>> = clusters
        .into_par_iter()
        .map(|cluster| fine_split_one_cluster(kmer_vecs, cluster, params, 0))
        .collect::<Result<Vec<_>>>()?;
    Ok(nested.into_iter().flatten().collect())
}

#[cfg(not(feature = "parallel"))]
fn fine_split_clusters(kmer_vecs: &[Vec<u16>], clusters: Vec<Vec<usize>>, params: &RadParams) -> Result<Vec<Vec<usize>>> {
    let mut out = Vec::new();
    for cluster in clusters {
        out.extend(fine_split_one_cluster(kmer_vecs, cluster, params, 0)?);
    }
    Ok(out)
}

/// Julia-RAD-style fine refinement: within each rough cluster, identify k-mer count
/// dimensions with high within-cluster variance, project reads onto those informative
/// dimensions, and run a second DP-means split. The recursion stops when there are no
/// informative k-mers, the cluster is too small, or max depth is reached.
fn fine_split_one_cluster(
    kmer_vecs: &[Vec<u16>],
    members: Vec<usize>,
    params: &RadParams,
    depth: usize,
) -> Result<Vec<Vec<usize>>> {
    if depth >= params.fine_split_max_depth || members.len() < params.min_cluster_size.saturating_mul(2).max(2) {
        return Ok(vec![members]);
    }

    let features = high_variance_kmers(kmer_vecs, &members, params);
    if features.len() < params.fine_split_min_features {
        return Ok(vec![members]);
    }

    let projected: Vec<Vec<u16>> = members
        .iter()
        .map(|&idx| features.iter().map(|&f| kmer_vecs[idx][f]).collect())
        .collect();

    let dp_params = DpMeansParams { radius: params.fine_split_radius, max_iter: params.fine_split_max_iter.max(1) };
    let local_clusters = dp_means(&projected, &dp_params, |c, v| normalized_sqeuclidean_projected(c, v, params.k));

    let mut mapped: Vec<Vec<usize>> = local_clusters
        .into_iter()
        .map(|c| c.members.into_iter().map(|local_i| members[local_i]).collect::<Vec<_>>())
        .collect();

    // Avoid over-splitting on noise: require at least two subclusters with enough support.
    let supported = mapped.iter().filter(|m| m.len() >= params.min_cluster_size).count();
    if supported < 2 {
        return Ok(vec![members]);
    }

    // Keep very small outlier groups with their nearest supported group instead of dropping reads.
    absorb_small_fine_split_groups(kmer_vecs, &mut mapped, params);

    let mut refined = Vec::new();
    for sub in mapped.into_iter().filter(|m| !m.is_empty()) {
        if sub.len() >= params.min_cluster_size.saturating_mul(2).max(2) {
            refined.extend(fine_split_one_cluster(kmer_vecs, sub, params, depth + 1)?);
        } else {
            refined.push(sub);
        }
    }
    Ok(refined)
}

fn high_variance_kmers(kmer_vecs: &[Vec<u16>], members: &[usize], params: &RadParams) -> Vec<usize> {
    if members.len() < 2 || kmer_vecs.is_empty() { return Vec::new(); }
    let dim = kmer_vecs[0].len();
    let n = members.len() as f64;
    let mut sum = vec![0.0f64; dim];
    let mut sumsq = vec![0.0f64; dim];

    for &idx in members {
        for (j, &x) in kmer_vecs[idx].iter().enumerate() {
            let xf = x as f64;
            sum[j] += xf;
            sumsq[j] += xf * xf;
        }
    }

    let mut scored = Vec::new();
    for j in 0..dim {
        if params.fine_split_drop_homopolymer_kmers && is_homopolymer_kmer_code(j as u64, params.k) {
            continue;
        }
        let mean = sum[j] / n;
        if mean <= 0.0 { continue; }
        let var = (sumsq[j] / n) - mean * mean;
        if var >= params.fine_split_min_var {
            // Variance-to-mean favors consistent allele-defining k-mers over random high-count repeats.
            let score = var / (mean + 1.0);
            scored.push((j, score));
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(params.fine_split_top_kmers.max(params.fine_split_min_features));
    scored.into_iter().map(|(j, _)| j).collect()
}

fn normalized_sqeuclidean_projected(c: &[f64], v: &[u16], k: usize) -> f64 {
    let mut sq = 0.0;
    let mut sc = 0.0;
    let mut sv = 0.0;
    for (&x, &y) in c.iter().zip(v) {
        let yy = y as f64;
        let d = x - yy;
        sq += d * d;
        sc += x;
        sv += yy;
    }
    if sc + sv == 0.0 { 0.0 } else { sq / (k as f64 * (sc + sv)) }
}

fn absorb_small_fine_split_groups(kmer_vecs: &[Vec<u16>], groups: &mut Vec<Vec<usize>>, params: &RadParams) {
    if groups.len() <= 1 { return; }
    let supported: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| if g.len() >= params.min_cluster_size { Some(i) } else { None })
        .collect();
    if supported.len() < 2 { return; }

    let centroids = group_centroids(kmer_vecs, groups);
    let mut moves = Vec::new();
    for (gidx, g) in groups.iter().enumerate() {
        if g.len() >= params.min_cluster_size { continue; }
        for &read_idx in g {
            let mut best = supported[0];
            let mut best_d = f64::INFINITY;
            for &target in &supported {
                let d = normalized_sqeuclidean_centroid(&centroids[target], &kmer_vecs[read_idx], params.k);
                if d < best_d { best_d = d; best = target; }
            }
            moves.push((gidx, read_idx, best));
        }
    }
    for (src, read_idx, dst) in moves {
        if src != dst { groups[dst].push(read_idx); }
    }
    groups.retain(|g| g.len() >= params.min_cluster_size);
}

fn group_centroids(kmer_vecs: &[Vec<u16>], groups: &[Vec<usize>]) -> Vec<Vec<f64>> {
    let dim = kmer_vecs.first().map_or(0, |v| v.len());
    groups
        .iter()
        .map(|g| {
            let mut c = vec![0.0; dim];
            if g.is_empty() { return c; }
            for &idx in g {
                for (j, &x) in kmer_vecs[idx].iter().enumerate() { c[j] += x as f64; }
            }
            let inv = 1.0 / g.len() as f64;
            for x in &mut c { *x *= inv; }
            c
        })
        .collect()
}

fn is_homopolymer_kmer_code(mut code: u64, k: usize) -> bool {
    if k == 0 { return false; }
    let first = code & 0b11;
    for _ in 1..k {
        code >>= 2;
        if (code & 0b11) != first { return false; }
    }
    true
}

#[cfg(feature = "parallel")]
fn compute_read_counts(reads: &[ReadRecord], params: &RadParams) -> Result<Vec<Vec<u16>>> {
    reads
        .par_iter()
        .map(|r| dense_counts(&r.seq, params.k, params.canonical))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn compute_read_counts(reads: &[ReadRecord], params: &RadParams) -> Result<Vec<Vec<u16>>> {
    reads.iter().map(|r| dense_counts(&r.seq, params.k, params.canonical)).collect()
}

#[cfg(feature = "parallel")]
fn compute_read_sketches(reads: &[ReadRecord], params: &RadParams) -> Result<Option<Vec<Vec<u16>>>> {
    if !params.use_sketch { return Ok(None); }
    let sk = KmerUtilsOptDensSketcher::new(params.k, params.sketch_size).canonical(params.canonical);
    reads
        .par_iter()
        .map(|r| sk.sketch(&r.seq).map(|s| s.sig))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

#[cfg(not(feature = "parallel"))]
fn compute_read_sketches(reads: &[ReadRecord], params: &RadParams) -> Result<Option<Vec<Vec<u16>>>> {
    if !params.use_sketch { return Ok(None); }
    let sk = KmerUtilsOptDensSketcher::new(params.k, params.sketch_size).canonical(params.canonical);
    reads
        .iter()
        .map(|r| sk.sketch(&r.seq).map(|s| s.sig))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

#[cfg(feature = "parallel")]
fn consensus_templates(reads: &[ReadRecord], clusters: &[Vec<usize>], params: &RadParams) -> Result<Vec<RadTemplate>> {
    clusters
        .par_iter()
        .map(|members| {
            let cluster_reads: Vec<_> = members.iter().map(|&i| reads[i].seq.clone()).collect();
            let seq = consensus_from_cluster(&cluster_reads, &params.consensus)?;
            Ok(RadTemplate { seq, cluster_size: cluster_abundance(reads, members) as usize, assigned_count: 0 })
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn consensus_templates(reads: &[ReadRecord], clusters: &[Vec<usize>], params: &RadParams) -> Result<Vec<RadTemplate>> {
    clusters
        .iter()
        .map(|members| {
            let cluster_reads: Vec<_> = members.iter().map(|&i| reads[i].seq.clone()).collect();
            let seq = consensus_from_cluster(&cluster_reads, &params.consensus)?;
            Ok(RadTemplate { seq, cluster_size: cluster_abundance(reads, members) as usize, assigned_count: 0 })
        })
        .collect()
}

#[cfg(feature = "parallel")]
fn compute_template_counts(templates: &[RadTemplate], params: &RadParams) -> Result<Vec<Vec<u16>>> {
    templates
        .par_iter()
        .map(|t| dense_counts(&t.seq, params.k, params.canonical))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn compute_template_counts(templates: &[RadTemplate], params: &RadParams) -> Result<Vec<Vec<u16>>> {
    templates.iter().map(|t| dense_counts(&t.seq, params.k, params.canonical)).collect()
}

#[cfg(feature = "parallel")]
fn compute_template_sketches(templates: &[RadTemplate], params: &RadParams) -> Result<Option<Vec<Vec<u16>>>> {
    if !params.use_sketch { return Ok(None); }
    let sk = KmerUtilsOptDensSketcher::new(params.k, params.sketch_size).canonical(params.canonical);
    templates
        .par_iter()
        .map(|t| sk.sketch(&t.seq).map(|s| s.sig))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

#[cfg(not(feature = "parallel"))]
fn compute_template_sketches(templates: &[RadTemplate], params: &RadParams) -> Result<Option<Vec<Vec<u16>>>> {
    if !params.use_sketch { return Ok(None); }
    let sk = KmerUtilsOptDensSketcher::new(params.k, params.sketch_size).canonical(params.canonical);
    templates
        .iter()
        .map(|t| sk.sketch(&t.seq).map(|s| s.sig))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

#[cfg(feature = "parallel")]
fn assign_reads_to_templates(
    read_counts: &[Vec<u16>],
    read_sketches: Option<&[Vec<u16>]>,
    tmpl_counts: &[Vec<u16>],
    tmpl_sketches: Option<&[Vec<u16>]>,
    params: &RadParams,
) -> Vec<usize> {
    read_counts
        .par_iter()
        .enumerate()
        .map(|(i, kv)| nearest_template_sketch_prefiltered(i, kv, read_sketches, tmpl_counts, tmpl_sketches, params))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn assign_reads_to_templates(
    read_counts: &[Vec<u16>],
    read_sketches: Option<&[Vec<u16>]>,
    tmpl_counts: &[Vec<u16>],
    tmpl_sketches: Option<&[Vec<u16>]>,
    params: &RadParams,
) -> Vec<usize> {
    read_counts
        .iter()
        .enumerate()
        .map(|(i, kv)| nearest_template_sketch_prefiltered(i, kv, read_sketches, tmpl_counts, tmpl_sketches, params))
        .collect()
}

fn nearest_template_sketch_prefiltered(
    read_idx: usize,
    kv: &[u16],
    read_sketches: Option<&[Vec<u16>]>,
    tmpl_counts: &[Vec<u16>],
    tmpl_sketches: Option<&[Vec<u16>]>,
    params: &RadParams,
) -> usize {
    if params.use_sketch {
        if let (Some(rs), Some(ts)) = (read_sketches, tmpl_sketches) {
            let candidates = sketch_candidates(
                &rs[read_idx],
                ts,
                params.k,
                params.sketch_prefilter_threshold,
                params.sketch_top_k,
            );
            if !candidates.is_empty() {
                return nearest_template_from_candidates(kv, tmpl_counts, params.k, &candidates);
            }
            if !params.sketch_fallback_exact {
                // If the user disables fallback and the filter found nothing, use all candidates
                // rather than panicking; this preserves a valid assignment.
                return nearest_template(kv, tmpl_counts, params.k);
            }
        }
    }
    nearest_template(kv, tmpl_counts, params.k)
}

fn sketch_candidates(
    query: &[u16],
    tmpl_sketches: &[Vec<u16>],
    k: usize,
    max_dist: f64,
    top_k: usize,
) -> Vec<usize> {
    let dist = SketchDistance::Bindash { k };
    let kkeep = top_k.max(1).min(tmpl_sketches.len().max(1));
    let mut best: Vec<(usize, f64)> = Vec::with_capacity(kkeep);

    for (idx, sig) in tmpl_sketches.iter().enumerate() {
        let d = dist.eval(query, sig);
        if best.len() < kkeep {
            best.push((idx, d));
            if best.len() == kkeep {
                best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else if d < best[kkeep - 1].1 {
            best[kkeep - 1] = (idx, d);
            best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    // Prefer candidates under the user cutoff, but never return an empty candidate list just
    // because the cutoff was too strict: falling back to all exact template distances is often
    // the real performance killer for large RAD runs.
    let under: Vec<usize> = best.iter().filter_map(|&(idx, d)| if d <= max_dist { Some(idx) } else { None }).collect();
    if under.is_empty() {
        best.into_iter().map(|(idx, _)| idx).collect()
    } else {
        under
    }
}

fn nearest_template_from_candidates(kv: &[u16], tmpl_counts: &[Vec<u16>], k: usize, candidates: &[usize]) -> usize {
    let mut best = candidates[0];
    let mut best_d = f64::INFINITY;
    for &tidx in candidates {
        let d = corrected_kmer_dist_full(kv, &tmpl_counts[tidx], k);
        if d < best_d { best_d = d; best = tidx; }
    }
    best
}

fn nearest_template(kv: &[u16], tmpl_counts: &[Vec<u16>], k: usize) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (tidx, tk) in tmpl_counts.iter().enumerate() {
        let d = corrected_kmer_dist_full(kv, tk, k);
        if d < best_d { best_d = d; best = tidx; }
    }
    best
}
