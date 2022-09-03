pub mod back_propagation;
pub mod calculate;
pub mod cost;
pub mod sizes;
use std::ptr::write_unaligned;

use rand::prelude::*;
use sizes::*;

#[derive(Debug)]
pub struct Network<const SIZES: &'static [usize]>
where
    [(); num_weights(SIZES)]:,
    [(); num_nodes(SIZES)]:,
    [(); SIZES.len()]:,
{
    pub weights: [f64; num_weights(SIZES)],
    biases: [f64; num_nodes(SIZES)],

    weight_costs: [f64; num_weights(SIZES)],
    bias_costs: [f64; num_nodes(SIZES)],
    node_values: [f64; num_nodes(SIZES)],

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
        let mut n = Network::<SIZES> {
            weights: [0.; num_weights(SIZES)],
            biases: [0.; num_nodes(SIZES)],

            weight_costs: [1.; num_weights(SIZES)],
            bias_costs: [1.; num_nodes(SIZES)],
            node_values: [1.; num_nodes(SIZES)],

            layer_indices: Self::calc_layer_indices(),
        };

        (n.weights, n.biases) = n.random_weights_and_biases();
        n
    }

    fn random_weights_and_biases(&self) -> ([f64; num_weights(SIZES)], [f64; num_nodes(SIZES)]) {
        let mut weights = [0.; num_weights(SIZES)];
        let mut biases = [0.; num_nodes(SIZES)];
        let mut rng = thread_rng();

        for layer in 1..(SIZES.len()) {
            for node in 0..idx(SIZES, layer) {
                for input in 0..idx(SIZES, layer - 1) {
                    weights[self.get_weight_index(layer, node, input)] =
                        rng.gen::<f64>() / (idx(SIZES, layer - 1) as f64).sqrt();
                }

                // biases[self.get_node_index(layer, node)] = rng.gen();
            }
        }

        (weights, biases)
    }

    pub const fn calc_layer_indices() -> [(usize, usize); SIZES.len()] {
        let mut arr = [(0, 0); SIZES.len()];
        let mut i = 1;
        while i < SIZES.len() {
            // Biases
            arr[i].0 = arr[i - 1].0 + SIZES[i - 1];
            // Weights
            if i == 1 {
                arr[i].1 = 0;
            } else {
                arr[i].1 = arr[i - 1].1 + SIZES[i - 1] * SIZES[i - 2];
            }
            i += 1;
        }

        arr
    }

    // No these aren't safe but ill be safe I promise 😇
    #[inline]
    pub fn get_node_index(&self, layer: usize, node: usize) -> usize {
        self.layer_indices[layer].0 + node
    }
    #[inline]
    pub fn get_weight_index(&self, layer: usize, node: usize, input: usize) -> usize {
        // Where the layer starts   +  number of connections per node   * node index + input index
        self.layer_indices[layer].1 + (SIZES[layer - 1]) * node + input
    }
}
