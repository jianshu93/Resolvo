#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Cluster {
    pub members: Vec<usize>,
    pub centroid: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct DpMeansParams {
    pub radius: f64,
    pub max_iter: usize,
}

impl Default for DpMeansParams {
    fn default() -> Self { Self { radius: 0.01, max_iter: 30 } }
}

/// DP-means clustering.
///
/// The expensive assignment step is parallelized. New-cluster creation is applied
/// deterministically after the assignment pass, preserving stable input order for
/// outliers. This is much faster than the original serial research-code loop while
/// retaining DP-means behavior for Resolvo's rough RAD clustering.
pub fn dp_means<F>(vectors: &[Vec<u16>], params: &DpMeansParams, dist: F) -> Vec<Cluster>
where
    F: Fn(&[f64], &[u16]) -> f64 + Sync,
{
    if vectors.is_empty() { return Vec::new(); }
    let dim = vectors[0].len();
    let mut centroids = vec![vectors[0].iter().map(|&x| x as f64).collect::<Vec<_>>()];
    let mut assignments = vec![usize::MAX; vectors.len()];

    for _ in 0..params.max_iter {
        let nearest = assign_all(vectors, &centroids, &dist);
        let mut changed = false;
        let mut new_centroid_for = Vec::new();

        for (i, &(best, best_d)) in nearest.iter().enumerate() {
            let a = if best_d > params.radius {
                let idx = centroids.len() + new_centroid_for.len();
                new_centroid_for.push(i);
                idx
            } else {
                best
            };
            if assignments[i] != a { changed = true; }
            assignments[i] = a;
        }

        for &i in &new_centroid_for {
            centroids.push(vectors[i].iter().map(|&x| x as f64).collect());
        }

        recompute_centroids_in_place(vectors, &assignments, &mut centroids, dim);
        if !changed { break; }
    }

    let mut members = vec![Vec::new(); centroids.len()];
    for (i, &a) in assignments.iter().enumerate() { members[a].push(i); }
    centroids
        .into_iter()
        .zip(members)
        .filter(|(_, m)| !m.is_empty())
        .map(|(centroid, members)| Cluster { centroid, members })
        .collect()
}

#[cfg(feature = "parallel")]
fn assign_all<F>(vectors: &[Vec<u16>], centroids: &[Vec<f64>], dist: &F) -> Vec<(usize, f64)>
where
    F: Fn(&[f64], &[u16]) -> f64 + Sync,
{
    vectors
        .par_iter()
        .map(|v| nearest_centroid(v, centroids, dist))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn assign_all<F>(vectors: &[Vec<u16>], centroids: &[Vec<f64>], dist: &F) -> Vec<(usize, f64)>
where
    F: Fn(&[f64], &[u16]) -> f64 + Sync,
{
    vectors.iter().map(|v| nearest_centroid(v, centroids, dist)).collect()
}

fn nearest_centroid<F>(v: &[u16], centroids: &[Vec<f64>], dist: &F) -> (usize, f64)
where
    F: Fn(&[f64], &[u16]) -> f64 + Sync,
{
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (cidx, c) in centroids.iter().enumerate() {
        let d = dist(c, v);
        if d < best_d { best_d = d; best = cidx; }
    }
    (best, best_d)
}

/// Recompute centroids without allocating any extra `k x dim` buffers.
///
/// The old `parallel` version used Rayon `fold`/`reduce` with one full `k x dim`
/// matrix per worker thread. With many clusters and 4096-dimensional k-mer
/// vectors, that can allocate tens to hundreds of GB.
///
/// This function keeps the same DP-means update rule:
///
/// - non-empty clusters are replaced by the arithmetic mean of their members;
/// - empty clusters keep their previous centroid, matching the old behavior.
///
/// The expensive nearest-centroid assignment step is still parallel when the
/// `parallel` feature is enabled. The centroid recomputation is deliberately
/// serial and in-place because it is memory-bandwidth bound and should not create
/// per-thread dense accumulation matrices.
fn recompute_centroids_in_place(
    vectors: &[Vec<u16>],
    assignments: &[usize],
    centroids: &mut [Vec<f64>],
    dim: usize,
) {
    let k = centroids.len();
    let mut counts = vec![0usize; k];

    for &a in assignments {
        debug_assert!(a < k);
        counts[a] += 1;
    }

    // Only clear centroids for clusters that have members. Empty clusters must
    // keep their previous centroid to preserve the old DP-means behavior.
    for (c, centroid) in centroids.iter_mut().enumerate() {
        if counts[c] > 0 {
            centroid.fill(0.0);
        }
    }

    for (v, &a) in vectors.iter().zip(assignments) {
        let centroid = &mut centroids[a];
        for j in 0..dim {
            centroid[j] += v[j] as f64;
        }
    }

    for (c, centroid) in centroids.iter_mut().enumerate() {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f64;
            for x in centroid.iter_mut() {
                *x *= inv;
            }
        }
    }
}

pub fn normalized_sqeuclidean_centroid(c: &[f64], v: &[u16], k: usize) -> f64 {
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
