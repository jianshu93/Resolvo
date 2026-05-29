use crate::dna2bit::Dna2Bit;
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct ReadRecord {
    pub id: String,
    pub seq: Dna2Bit,
    pub qual: Option<Vec<u8>>,
    /// Number of original reads represented by this record.
    /// FASTA headers like `>Uniq1;size=48;` set this to 48; ordinary FASTQ records use 1.
    pub count: u64,
}

#[derive(Clone, Debug)]
pub struct DerepRecord {
    pub seq: Dna2Bit,
    pub count: u64,
    pub read_indices: Vec<usize>,
}

#[cfg(not(feature = "parallel"))]
pub fn dereplicate(reads: &[ReadRecord]) -> Vec<DerepRecord> {
    let mut map: HashMap<Dna2Bit, (u64, Vec<usize>)> = HashMap::new();
    for (i, r) in reads.iter().enumerate() {
        let e = map.entry(r.seq.clone()).or_insert_with(|| (0, Vec::new()));
        e.0 += r.count.max(1);
        e.1.push(i);
    }
    finish_derep_map(map)
}

#[cfg(feature = "parallel")]
pub fn dereplicate(reads: &[ReadRecord]) -> Vec<DerepRecord> {
    // Parallel fold/reduce avoids one large contended mutex while keeping exact dereplication.
    let map = reads
        .par_iter()
        .enumerate()
        .fold(HashMap::new, |mut local: HashMap<Dna2Bit, (u64, Vec<usize>)>, (i, r)| {
            let e = local.entry(r.seq.clone()).or_insert_with(|| (0, Vec::new()));
            e.0 += r.count.max(1);
            e.1.push(i);
            local
        })
        .reduce(HashMap::new, |mut a, b| {
            for (seq, (count, mut idxs)) in b {
                let e = a.entry(seq).or_insert_with(|| (0, Vec::new()));
                e.0 += count;
                e.1.append(&mut idxs);
            }
            a
        });
    finish_derep_map(map)
}

fn finish_derep_map(map: HashMap<Dna2Bit, (u64, Vec<usize>)>) -> Vec<DerepRecord> {
    let mut out: Vec<_> = map
        .into_iter()
        .map(|(seq, (count, mut read_indices))| {
            read_indices.sort_unstable();
            DerepRecord { seq, count, read_indices }
        })
        .collect();
    out.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.seq.len().cmp(&b.seq.len())));
    out
}
