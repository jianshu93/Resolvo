use crate::distance::{anndists_hamming_u16, bindash_distance_from_hamming};

#[derive(Clone, Copy, Debug)]
pub enum SketchDistance {
    Hamming,
    Bindash { k: usize },
}

#[inline]
pub fn sketch_hamming(a: &[u16], b: &[u16]) -> f64 {
    anndists_hamming_u16(a, b)
}

#[inline]
pub fn sketch_bindash_distance(a: &[u16], b: &[u16], k: usize) -> f64 {
    bindash_distance_from_hamming(sketch_hamming(a, b), k)
}

impl SketchDistance {
    pub fn eval(self, a: &[u16], b: &[u16]) -> f64 {
        match self {
            SketchDistance::Hamming => sketch_hamming(a, b),
            SketchDistance::Bindash { k } => sketch_bindash_distance(a, b, k),
        }
    }
}
