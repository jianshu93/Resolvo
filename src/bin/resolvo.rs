use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use resolvo::algorithms::fad::{fad_denoise, FadParams};
use resolvo::algorithms::rad::{rad_denoise, RadClusterMode, RadParams};
use resolvo::consensus::ConsensusParams;
use resolvo::io::{read_fasta_fastq, write_fasta};

#[derive(Parser, Debug)]
#[command(name = "resolvo")]
#[command(about = "Long-amplicon denoising by template recovery from noisy long-read clusters")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliRadClusterMode {
    /// Faithful/reference dense-kmer DP-means over all reads.
    Exact,
    /// Loose OptDens/Bindash preclustering followed by exact dense-kmer DP-means inside each bin.
    SketchPrecluster,
    /// OptDens/Bindash medoid DP-means only. Fastest but approximate.
    SketchOnly,
}

impl From<CliRadClusterMode> for RadClusterMode {
    fn from(value: CliRadClusterMode) -> Self {
        match value {
            CliRadClusterMode::Exact => RadClusterMode::Exact,
            CliRadClusterMode::SketchPrecluster => RadClusterMode::SketchPreclusterThenExact,
            CliRadClusterMode::SketchOnly => RadClusterMode::SketchOnly,
        }
    }
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// FAD: abundance-ordered template selection.
    Fad {
        #[arg(short, long)] input: String,
        #[arg(short, long)] output: String,
        #[arg(long, default_value_t = 6)] k: usize,
        #[arg(long, default_value_t = 1.0)] neighbor_threshold: f64,
        #[arg(long, default_value_t = 2)] min_count: u64,
        #[arg(long, default_value_t = 512)] sketch_size: usize,
        #[arg(long, default_value_t = true)] use_sketch: bool,
        #[arg(long, default_value_t = 0.03)] sketch_prefilter_threshold: f64,
    },
    /// RAD: k-mer-space clustering followed by consensus reconstruction.
    Rad {
        #[arg(short, long)] input: String,
        #[arg(short, long)] output: String,
        #[arg(long, default_value_t = 6)] k: usize,
        #[arg(long, default_value_t = 0.01)] rough_radius: f64,
        #[arg(long, default_value_t = 5)] min_cluster_size: usize,
        #[arg(long, default_value_t = 1)] polish_rounds: usize,
        #[arg(long, default_value_t = true)] fad_clean: bool,
        /// Enable recursive high-variance k-mer fine splitting inside rough RAD clusters.
        #[arg(long, default_value_t = true)] fine_split: bool,
        /// Maximum recursive fine-splitting depth per rough cluster.
        #[arg(long, default_value_t = 4)] fine_split_max_depth: usize,
        /// Number of high-variance k-mer dimensions used for each fine split.
        #[arg(long, default_value_t = 128)] fine_split_top_kmers: usize,
        /// Minimum within-cluster k-mer variance needed to use a k-mer for fine splitting.
        #[arg(long, default_value_t = 0.05)] fine_split_min_var: f64,
        /// DP-means radius on the projected high-variance k-mer subspace.
        #[arg(long, default_value_t = 0.01)] fine_split_radius: f64,
        /// Minimum number of high-variance k-mers needed before a fine split is attempted.
        #[arg(long, default_value_t = 8)] fine_split_min_features: usize,

        /// RAD clustering backend. Use exact for validation, sketch-precluster for large full-operon datasets.
        #[arg(long, value_enum, default_value = "exact")]
        cluster_mode: CliRadClusterMode,
        /// Enable OptDens/Bindash sketches for RAD-supported stages.
        #[arg(long, default_value_t = true)] use_sketch: bool,
        /// OptDens signature size.
        #[arg(long, default_value_t = 512)] sketch_size: usize,
        /// Loose Bindash-distance cutoff for candidate prefiltering.
        #[arg(long, default_value_t = 0.03)] sketch_prefilter_threshold: f64,
        /// Top-K approximate sketch candidates to exact-check during final reassignment.
        #[arg(long, default_value_t = 32)] sketch_top_k: usize,
        /// Fall back to full exact scan if the sketch filter returns no candidates.
        #[arg(long, default_value_t = true)] sketch_fallback_exact: bool,
        /// Sketch DP-means/precluster radius in Bindash-distance units.
        #[arg(long, default_value_t = 0.03)] sketch_radius: f64,
        /// Maximum iterations for sketch medoid clustering/preclustering.
        #[arg(long, default_value_t = 8)] sketch_max_iter: usize,
        /// Ends-free Needleman-Wunsch band radius used during consensus polishing.
        /// For divergent full-length/operon reads, try 64-128.
        /// Set to -1 for full unbanded NW.
        #[arg(long, default_value_t = 64)] align_band: i32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Fad { input, output, k, neighbor_threshold, min_count, sketch_size, use_sketch, sketch_prefilter_threshold } => {
            let reads = read_fasta_fastq(&input)?;
            let params = FadParams { k, neighbor_threshold, min_count, sketch_size, use_sketch, sketch_prefilter_threshold, ..FadParams::default() };
            let res = fad_denoise(&reads, &params)?;
            let records = res.templates.iter().enumerate().map(|(i, t)| {
                let u = &res.unique[t.unique_index];
                (format!("ResolvoFAD_{} size={} assigned={}", i + 1, t.count, t.assigned_count), u.seq.clone())
            });
            write_fasta(output, records)?;
        }
        Commands::Rad {
            input,
            output,
            k,
            rough_radius,
            min_cluster_size,
            polish_rounds,
            fad_clean,
            fine_split,
            fine_split_max_depth,
            fine_split_top_kmers,
            fine_split_min_var,
            fine_split_radius,
            fine_split_min_features,
            cluster_mode,
            use_sketch,
            sketch_size,
            sketch_prefilter_threshold,
            sketch_top_k,
            sketch_fallback_exact,
            sketch_radius,
            sketch_max_iter,
            align_band,
        } => {
            let reads = read_fasta_fastq(&input)?;
            let mut align = resolvo::align::AlignParams::default();
            align.band = align_band;
            let consensus = ConsensusParams { k, polish_rounds, align, ..ConsensusParams::default() };
            let fad = FadParams {
                k,
                min_count: 1,
                use_sketch,
                sketch_size,
                sketch_prefilter_threshold,
                ..FadParams::default()
            };
            let params = RadParams {
                k,
                rough_radius,
                min_cluster_size,
                consensus,
                run_fad_clean: fad_clean,
                fad,
                fine_split,
                fine_split_max_depth,
                fine_split_top_kmers,
                fine_split_min_var,
                fine_split_radius,
                fine_split_min_features,
                fine_split_drop_homopolymer_kmers: true,
                cluster_mode: cluster_mode.into(),
                use_sketch,
                sketch_size,
                sketch_prefilter_threshold,
                sketch_top_k,
                sketch_fallback_exact,
                sketch_radius,
                sketch_max_iter,
                ..RadParams::default()
            };
            let res = rad_denoise(&reads, &params)?;
            let records = res.templates.iter().enumerate().map(|(i, t)| {
                (format!("ResolvoRAD_{} cluster_size={} assigned={}", i + 1, t.cluster_size, t.assigned_count), t.seq.clone())
            });
            write_fasta(output, records)?;
        }
    }
    Ok(())
}
