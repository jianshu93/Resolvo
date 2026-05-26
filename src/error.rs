use thiserror::Error;

#[derive(Debug, Error)]
pub enum ResolvoError {
    #[error("invalid DNA base {base:?} at position {pos}")]
    InvalidBase { base: u8, pos: usize },

    #[error("sequence is shorter than k={k}: length={len}")]
    SequenceTooShort { len: usize, k: usize },

    #[error("k must be in 1..=15 for u64 k-mer codes, got {0}")]
    InvalidK(usize),

    #[error("input contains no reads")]
    EmptyInput,

    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ResolvoError>;
