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

        let k = centroids.len();
        let (sums, counts) = compute_sums(vectors, &assignments, k, dim);
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                centroids[c] = sums[c].iter().map(|x| x * inv).collect();
            }
        }
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

#[cfg(feature = "parallel")]
fn compute_sums(vectors: &[Vec<u16>], assignments: &[usize], k: usize, dim: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    vectors
        .par_iter()
        .zip(assignments.par_iter().copied())
        .fold(
            || (vec![vec![0.0; dim]; k], vec![0usize; k]),
            |mut acc, (v, a)| {
                acc.1[a] += 1;
                for j in 0..dim { acc.0[a][j] += v[j] as f64; }
                acc
            },
        )
        .reduce(
            || (vec![vec![0.0; dim]; k], vec![0usize; k]),
            |mut a, b| {
                for c in 0..k {
                    a.1[c] += b.1[c];
                    for j in 0..dim { a.0[c][j] += b.0[c][j]; }
                }
                a
            },
        )
}

#[cfg(not(feature = "parallel"))]
fn compute_sums(vectors: &[Vec<u16>], assignments: &[usize], k: usize, dim: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut sums = vec![vec![0.0; dim]; k];
    let mut counts = vec![0usize; k];
    for (v, &a) in vectors.iter().zip(assignments) {
        counts[a] += 1;
        for j in 0..dim { sums[a][j] += v[j] as f64; }
    }
    (sums, counts)
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
