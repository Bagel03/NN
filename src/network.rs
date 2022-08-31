mod back_propagation;
mod calculate;
mod cost;
pub mod sizes;
use sizes::*;

pub struct Network<const SIZES: &'static [usize]>
where
    [(); num_weights(SIZES)]:,
    [(); num_nodes(SIZES)]:,
    [(); SIZES.len()]:,
{
    weights: [f64; num_weights(SIZES)],
    biases: [f64; num_nodes(SIZES)],

    weight_costs: [f64; num_weights(SIZES)],
    bias_costs: [f64; num_nodes(SIZES)],

    // Lookup        bias   weight
    layer_indices: [(usize, usize); SIZES.len()],
}

impl<const SIZES: &'static [usize]> Network<SIZES>
where
    [(); num_weights(SIZES)]:,
    [(); num_nodes(SIZES)]:,
    [(); SIZES.len()]:,
{
    pub fn new() -> Network<SIZES> {
        Network::<SIZES> {
            weights: [0.; num_weights(SIZES)],
            biases: [0.; num_nodes(SIZES)],

            weight_costs: [0.; num_weights(SIZES)],
            bias_costs: [0.; num_nodes(SIZES)],

            layer_indices: Self::calc_layer_indices(),
        }
    }

    const fn calc_layer_indices() -> [(usize, usize); SIZES.len()] {
        let mut arr = [(0, 0); SIZES.len()];
        let mut i = 1;
        while i < SIZES.len() {
            arr[i].0 = arr[i - 1].0 + SIZES[i];
            arr[i].1 = arr[i - 1].1 + SIZES[i] * SIZES[i - 1];
            i += 1;
        }

        arr
    }
}
