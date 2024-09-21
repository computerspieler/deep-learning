#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]

pub mod matrix;
pub mod ai_trait;
pub mod dense;
pub mod activation;
pub mod dataset;
pub mod flatten2d;
pub mod vector;
pub mod convolution;
pub mod pooling;
pub mod sequential;

pub use ai_trait::AIModule;