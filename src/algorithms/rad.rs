use crate::algorithms::fad::{fad_denoise, FadParams};
use crate::consensus::{consensus_from_cluster, ConsensusParams};
use crate::derep::ReadRecord;
use crate::distance::corrected_kmer_dist_full;
use crate::dp_means::{dp_means, normalized_sqeuclidean_centroid, DpMeansParams};
use crate::error::{ResolvoError, Result};
use crate::kmer::dense_counts;
use crate::logging::{cluster_summary, log, StageTimer};
use crate::sketch::{KmerUtilsOptDensSketcher, SketchDistance};

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
    fn default() -> Self { Self::Exact }
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
            min_cluster_size: 5,
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
            cluster_mode: RadClusterMode::Exact,
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
        "RAD input reads={} k={} mode={:?} fine_split={} use_sketch={} sketch_size={} align_band={} max_consensus_reads={} rough_max_iter={}",
        reads.len(),
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
    log(params.verbose, format!("rough {}", cluster_summary(&clusters)));

    if clusters.is_empty() { clusters.push((0..reads.len()).collect()); }
    clusters.retain(|m| m.len() >= params.min_cluster_size);
    if clusters.is_empty() { clusters.push((0..reads.len()).collect()); }
    log(params.verbose, format!("after min_cluster_size filter {}", cluster_summary(&clusters)));

    if params.fine_split {
        let t = StageTimer::start("RAD recursive high-variance fine splitting", params.verbose);
        clusters = fine_split_clusters(&kmer_vecs, clusters, params)?;
        t.done();
        clusters.retain(|m| m.len() >= params.min_cluster_size);
        if clusters.is_empty() { clusters.push((0..reads.len()).collect()); }
        log(params.verbose, format!("fine-split {}", cluster_summary(&clusters)));
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
            .map(|(i, t)| ReadRecord { id: format!("rad_consensus_{i}"), seq: t.seq.clone(), qual: None })
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
    for &a in &read_assignments { templates[a].assigned_count += 1; }

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
            let bins = sketch_dp_means(sketches, params);
            exact_dp_within_bins(kmer_vecs, bins, params)
        }
        RadClusterMode::SketchOnly => {
            if !params.use_sketch {
                return exact_dp_clusters(kmer_vecs, params);
            }
            let sketches = sketches.ok_or_else(|| ResolvoError::InvalidParam("sketch-only clustering requested but sketches are unavailable".into()))?;
            let _ = reads; // keep signature symmetric; future versions may use read lengths/ids here.
            Ok(sketch_dp_means(sketches, params))
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

/// Sketch medoid DP-means. This is a rough clustering/preclustering method, not a replacement
/// for exact dense-kmer RAD unless `cluster_mode = SketchOnly` is explicitly selected.
fn sketch_dp_means(sketches: &[Vec<u16>], params: &RadParams) -> Vec<Vec<usize>> {
    if sketches.is_empty() { return Vec::new(); }
    let dist = SketchDistance::Bindash { k: params.k };
    let mut medoids = vec![0usize];
    let mut assignments = vec![usize::MAX; sketches.len()];

    for _ in 0..params.sketch_max_iter.max(1) {
        let nearest = assign_all_sketches(sketches, &medoids, dist);
        let mut changed = false;
        let mut new_medoids = Vec::new();

        for (i, &(best, best_d)) in nearest.iter().enumerate() {
            let a = if best_d > params.sketch_radius && medoids.len() + new_medoids.len() < params.sketch_max_bins.max(1) {
                let idx = medoids.len() + new_medoids.len();
                new_medoids.push(i);
                idx
            } else {
                best
            };
            if assignments[i] != a { changed = true; }
            assignments[i] = a;
        }
        medoids.extend(new_medoids);

        let mut members = vec![Vec::new(); medoids.len()];
        for (i, &a) in assignments.iter().enumerate() {
            if a < members.len() { members[a].push(i); }
        }
        recompute_medoids(sketches, &members, &mut medoids, dist);

        if !changed { break; }
    }

    let mut members = vec![Vec::new(); medoids.len()];
    for (i, &a) in assignments.iter().enumerate() {
        if a < members.len() { members[a].push(i); }
    }
    members.into_iter().filter(|m| !m.is_empty()).collect()
}

#[cfg(feature = "parallel")]
fn assign_all_sketches(sketches: &[Vec<u16>], medoids: &[usize], dist: SketchDistance) -> Vec<(usize, f64)> {
    sketches
        .par_iter()
        .map(|s| nearest_medoid(s, sketches, medoids, dist))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn assign_all_sketches(sketches: &[Vec<u16>], medoids: &[usize], dist: SketchDistance) -> Vec<(usize, f64)> {
    sketches.iter().map(|s| nearest_medoid(s, sketches, medoids, dist)).collect()
}

fn nearest_medoid(query: &[u16], sketches: &[Vec<u16>], medoids: &[usize], dist: SketchDistance) -> (usize, f64) {
    let mut best_cluster = 0usize;
    let mut best_d = f64::INFINITY;
    for (cidx, &midx) in medoids.iter().enumerate() {
        let d = dist.eval(query, &sketches[midx]);
        if d < best_d {
            best_d = d;
            best_cluster = cidx;
        }
    }
    (best_cluster, best_d)
}

#[cfg(feature = "parallel")]
fn recompute_medoids(sketches: &[Vec<u16>], members: &[Vec<usize>], medoids: &mut [usize], dist: SketchDistance) {
    let new_medoids: Vec<usize> = members
        .par_iter()
        .enumerate()
        .map(|(cidx, m)| best_medoid_for_cluster(sketches, m, medoids[cidx], dist))
        .collect();
    for (m, new_m) in medoids.iter_mut().zip(new_medoids) { *m = new_m; }
}

#[cfg(not(feature = "parallel"))]
fn recompute_medoids(sketches: &[Vec<u16>], members: &[Vec<usize>], medoids: &mut [usize], dist: SketchDistance) {
    for (cidx, m) in members.iter().enumerate() {
        medoids[cidx] = best_medoid_for_cluster(sketches, m, medoids[cidx], dist);
    }
}

fn best_medoid_for_cluster(sketches: &[Vec<u16>], members: &[usize], fallback: usize, dist: SketchDistance) -> usize {
    if members.is_empty() { return fallback; }
    // Full medoid search is O(m^2). Cap to a representative prefix for very large bins; this
    // keeps rough sketch preclustering cheap. Exact DP-means/consensus follows later.
    let sample_len = members.len().min(256);
    let sample = &members[..sample_len];
    let mut best = members[0];
    let mut best_sum = f64::INFINITY;
    for &candidate in sample {
        let mut sum = 0.0;
        for &other in sample {
            sum += dist.eval(&sketches[candidate], &sketches[other]);
        }
        if sum < best_sum {
            best_sum = sum;
            best = candidate;
        }
    }
    best
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
            Ok(RadTemplate { seq, cluster_size: members.len(), assigned_count: 0 })
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
            Ok(RadTemplate { seq, cluster_size: members.len(), assigned_count: 0 })
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
