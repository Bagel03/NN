use super::sizes::*;
use crate::activation::Activation;

impl<const SIZES: &'static [usize]> super::Network<SIZES>
where
    [(); num_weights(SIZES)]:,
    [(); num_nodes(SIZES)]:,
    [(); SIZES.len()]:,
{
    pub type outputs_and_weighted_inputs = ([f64; num_nodes(SIZES)], [f64; num_nodes(SIZES)]);
    // (outputs, weighted_inputs)
    pub fn calculate_all_with_weighted_inputs(
        &self,
        inputs: [f64; idx(SIZES, 0)],
    ) -> ([f64; num_nodes(SIZES)], [f64; num_nodes(SIZES)]) {
        let mut results = ([0.; num_nodes(SIZES)], [0.; num_nodes(SIZES)]);

        for layer in 0..SIZES.len() {
            for node in 0..idx(SIZES, layer) {
                if layer == 0 {
                    results.0[node] = inputs[node];
                    continue;
                }

                let node_index = self.get_node_index(layer, node);

                for input in 0..idx(SIZES, layer - 1) {
                    // Add the previous activation * weight to our current weighted sum
                    results.1[node_index] += results.0[self.get_node_index(layer - 1, input)]
                        * self.biases[self.get_weight_index(layer, node, input)];
                }

                results.0[node_index] =
                    Activation::function(results.1[node_index] + self.biases[node_index]);
            }
        }

        results
    }

    pub fn calculate_all(&self, inputs: [f64; idx(SIZES, 0)]) -> [f64; num_nodes(SIZES)] {
        let mut results = [0.; num_nodes(SIZES)];
        for layer in 0..SIZES.len() {
            for node in 0..idx(SIZES, layer) {
                if layer == 0 {
                    results[node] = inputs[node];
                    continue;
                }

                let node_index = self.get_node_index(layer, node);
                // Set it to the bias to begin
                results[node_index] = self.biases[node_index];

                for input in 0..idx(SIZES, layer - 1) {
                    results[node_index] += results[self.get_node_index(layer - 1, input)]
                        * self.biases[self.get_weight_index(layer, node, input)];
                }

                results[node_index] = Activation::function(results[node_index]);
            }
        }
        results
    }

    pub fn calculate_outputs(&self, inputs: [f64; idx(SIZES, 0)]) -> [f64; outputs(SIZES)] {
        let results = self.calculate_all(inputs);
        let mut outputs = [0.; outputs(SIZES)];
        for i in outputs.len()..0 {
            outputs[i] = results[num_nodes(SIZES) - i - 1];
        }
        outputs
    }
}
