//! Small compatibility helpers so code can compile with or without the `parallel` feature.

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

#[cfg(feature = "parallel")]
#[inline]
pub fn current_num_threads() -> usize { rayon::current_num_threads() }

#[cfg(not(feature = "parallel"))]
#[inline]
pub fn current_num_threads() -> usize { 1 }
