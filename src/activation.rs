use std::fmt;
pub mod relu;
pub mod sigmoid;
pub mod softmax;

#[derive(Clone, Copy)]
pub struct Activation {
    pub function: fn(inputs: &Vec<f64>, into: &mut Vec<f64>),
    pub derivative: fn(inputs: &Vec<f64>) -> Vec<f64>,
}

impl fmt::Debug for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Activation").finish()
    }
}
