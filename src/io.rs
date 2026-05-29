use crate::derep::ReadRecord;
use crate::dna2bit::Dna2Bit;
use crate::error::Result;
use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;


fn parse_size_from_header(header: &str) -> u64 {
    // Supports common dereplicated FASTA headers:
    //   >Uniq1;size=48;
    //   >Uniq1 size=48
    //   >Uniq1;size=48;foo=bar
    for tok in header.split(|c: char| c == ';' || c.is_ascii_whitespace()) {
        if let Some(v) = tok.strip_prefix("size=") {
            if let Ok(n) = v.parse::<u64>() { return n.max(1); }
        }
    }
    1
}

fn clean_id_from_header(header: &str) -> String {
    header
        .split(|c: char| c == ';' || c.is_ascii_whitespace())
        .next()
        .unwrap_or(header)
        .to_string()
}

fn open_reader<P: AsRef<Path>>(path: P) -> Result<Box<dyn BufRead>> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref)?;
    let reader: Box<dyn Read> = if path_ref.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::new(reader)))
}

pub fn read_fasta_fastq<P: AsRef<Path>>(path: P) -> Result<Vec<ReadRecord>> {
    let mut reader = open_reader(path)?;
    let mut first = String::new();
    reader.read_line(&mut first)?;
    if first.starts_with('>') { read_fasta_from_first(reader, first) }
    else if first.starts_with('@') { read_fastq_from_first(reader, first) }
    else { Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "expected FASTA or FASTQ").into()) }
}

fn read_fasta_from_first(mut reader: Box<dyn BufRead>, first: String) -> Result<Vec<ReadRecord>> {
    let mut records = Vec::new();
    let mut header0 = first[1..].trim().to_string();
    let mut id = clean_id_from_header(&header0);
    let mut count = parse_size_from_header(&header0);
    let mut seq = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 { break; }
        let l = line.trim_end();
        if let Some(rest) = l.strip_prefix('>') {
            if !seq.is_empty() {
                records.push(ReadRecord { id, seq: Dna2Bit::from_ascii(&seq)?, qual: None, count });
                seq.clear();
            }
            header0 = rest.trim().to_string();
            id = clean_id_from_header(&header0);
            count = parse_size_from_header(&header0);
        } else {
            seq.extend_from_slice(l.as_bytes());
        }
    }
    if !seq.is_empty() {
        records.push(ReadRecord { id, seq: Dna2Bit::from_ascii(&seq)?, qual: None, count });
    }
    Ok(records)
}

fn read_fastq_from_first(mut reader: Box<dyn BufRead>, first: String) -> Result<Vec<ReadRecord>> {
    let mut records = Vec::new();
    let mut header = first;
    loop {
        if !header.starts_with('@') { break; }
        let id = header[1..].trim().to_string();
        let mut seq = String::new();
        let mut plus = String::new();
        let mut qual = String::new();
        if reader.read_line(&mut seq)? == 0 { break; }
        reader.read_line(&mut plus)?;
        reader.read_line(&mut qual)?;
        let qual_vec = qual.trim_end().as_bytes().to_vec();
        records.push(ReadRecord { id, seq: Dna2Bit::from_ascii(seq.trim_end().as_bytes())?, qual: Some(qual_vec), count: 1 });
        header.clear();
        if reader.read_line(&mut header)? == 0 { break; }
    }
    Ok(records)
}

pub fn write_fasta<P: AsRef<Path>, I>(path: P, records: I) -> Result<()>
where I: IntoIterator<Item = (String, Dna2Bit)> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    for (id, seq) in records {
        writeln!(w, ">{id}")?;
        let s = seq.to_ascii();
        for chunk in s.chunks(80) { writeln!(w, "{}", String::from_utf8_lossy(chunk))?; }
    }
    Ok(())
}
