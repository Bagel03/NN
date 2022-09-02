pub mod back_propagation;
pub mod calculate;
pub mod cost;
pub mod sizes;
use rand::prelude::*;
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

            weight_costs: [0.; num_weights(SIZES)],
            bias_costs: [0.; num_nodes(SIZES)],
            node_values: [0.; num_nodes(SIZES)],

            layer_indices: Self::calc_layer_indices(),
        };

        n.weight_costs = n.random_weights();
        n
    }

    fn random_weights(&self) -> [f64; num_weights(SIZES)] {
        let mut results = [0.; num_weights(SIZES)];
        let mut rng = thread_rng();

        for layer in 0..(SIZES.len() - 1) {
            for input_node in 0..idx(SIZES, layer) {
                for output_node in 0..idx(SIZES, layer + 1) {
                    results[self.get_weight_index(layer, output_node, input_node)] = rng.gen();
                }
            }
        }
        results
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

    // No these aren't safe but ill be safe I promise 😇
    #[inline]
    pub fn get_node_index(&self, layer: usize, node: usize) -> usize {
        self.layer_indices[layer].0 + node
    }
    #[inline]
    pub fn get_weight_index(&self, layer: usize, node: usize, input: usize) -> usize {
        // Where the layer starts   +  number of connections per node   * node index + input index
        self.layer_indices[layer].1 + (SIZES[layer - 1] * SIZES[layer]) * node + input
    }
}
