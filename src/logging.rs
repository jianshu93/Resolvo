use std::time::Instant;

#[derive(Clone, Debug)]
pub struct StageTimer {
    name: &'static str,
    start: Instant,
    enabled: bool,
}

impl StageTimer {
    pub fn start(name: &'static str, enabled: bool) -> Self {
        if enabled {
            eprintln!("[resolvo] START {name}");
        }
        Self { name, start: Instant::now(), enabled }
    }

    pub fn done(self) {
        if self.enabled {
            eprintln!("[resolvo] DONE  {} in {:.3}s", self.name, self.start.elapsed().as_secs_f64());
        }
    }
}

#[inline]
pub fn log(enabled: bool, msg: impl AsRef<str>) {
    if enabled {
        eprintln!("[resolvo] {}", msg.as_ref());
    }
}

pub fn cluster_summary(clusters: &[Vec<usize>]) -> String {
    if clusters.is_empty() {
        return "clusters=0".to_string();
    }
    let mut sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();
    sizes.sort_unstable();
    let n = sizes.len();
    let sum: usize = sizes.iter().sum();
    let p50 = sizes[n / 2];
    let p90 = sizes[((n as f64 * 0.90).floor() as usize).min(n - 1)];
    let p99 = sizes[((n as f64 * 0.99).floor() as usize).min(n - 1)];
    format!(
        "clusters={} reads={} min={} p50={} p90={} p99={} max={}",
        n,
        sum,
        sizes[0],
        p50,
        p90,
        p99,
        sizes[n - 1]
    )
}
