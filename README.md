# Resolvo

**Resolvo** recovers true long-amplicon templates from noisy long-read clusters.

This crate is intentionally built for the high-performance purposes:

- 2-bit DNA storage for Resolvo's internal read representation.
- One Permutation Hashing MinHash sketches.
- Fast sketch Hamming comparisons.
- Ends-free Needleman-Wunsch alignment (SIMD) for consensus polishing.
- FAD: abundance-ordered template selection with sketch prefilter + exact corrected k-mer confirmation.
- RAD: k-mer-space DP-means rough clustering + alignment consensus + optional FAD-cleaning.

## Build

```bash
cargo build --release
```

## Run

```bash
resolvo fad --input reads.fasta --output templates.fasta --k 6 --sketch-size 512
resolvo rad --input reads.fasta --output templates.fasta --k 6 --rough-radius 0.01 --min-cluster-size 5
```



## Parallel execution

Resolvo enables Rayon by default through the `parallel` feature. The heavy stages are parallelized:

- exact dereplication via parallel fold/reduce maps
- dense k-mer vector construction
- kmerutils OptDens sketch construction
- FAD nearest-template reassignment
- DP-means assignment and centroid reduction
- RAD per-cluster consensus generation
- per-read alignment pileup construction within consensus polishing
- final read-to-template reassignment

## Notes
This is a performance-oriented crate. It reuses kmerutils/anndists and DADA2 style alignment.