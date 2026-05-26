use resolvo::derep::ReadRecord;
use resolvo::{Dna2Bit, FadParams, fad_denoise};

fn main() -> anyhow::Result<()> {
    let reads = vec![
        ReadRecord { id: "r1".into(), seq: Dna2Bit::from_ascii(b"ACGTACGTACGT")?, qual: None },
        ReadRecord { id: "r2".into(), seq: Dna2Bit::from_ascii(b"ACGTACGTACGT")?, qual: None },
        ReadRecord { id: "r3".into(), seq: Dna2Bit::from_ascii(b"ACGTACGTACGA")?, qual: None },
    ];
    let params = FadParams { k: 3, min_count: 1, ..FadParams::default() };
    let out = fad_denoise(&reads, &params)?;
    println!("templates: {}", out.templates.len());
    Ok(())
}
