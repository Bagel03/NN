use super::{cost::DataPoint, sizes::*};
use crate::activation::Activation;

impl<const SIZES: &'static [usize]> super::Network<SIZES>
where
    [(); num_weights(SIZES)]:,
    [(); num_nodes(SIZES)]:,
    [(); SIZES.len()]:,
    [(); idx(SIZES, 0)]:,
    [(); outputs(SIZES)]:,
{
    fn update_weights_and_biases(&mut self, learn_rate: f64) {
        for i in 0..self.weights.len() {
            self.weights[i] -= self.weight_costs[i] * learn_rate;
            if i < self.biases.len() {
                self.biases[i] -= self.bias_costs[i] * learn_rate;
            }
        }
    }

    fn calculate_output_derivatives(&self, node_index: usize, actual_output: f64) {}

    pub fn learn<const T: usize>(&mut self, data: [DataPoint<SIZES>; T], learn_rate: f64) {
        // Could make this multithreaded
        for data_point in data {
            let output = self.calculate_outputs(data_point.inputs);
            // Go backwards
            for layer in (1..SIZES.len()).rev() {
                for node in 0..layer {
                    if layer == SIZES.len() - 1 {
                        // Output layer needs cost derivative too
                        derivative *= self.cost_derivative(output[node], data_point.expected[node]);
                    }
                }
            }
        }

        self.update_weights_and_biases(learn_rate);
    }
}
