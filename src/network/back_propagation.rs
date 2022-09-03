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
        }
        for i in 0..self.biases.len() {
            self.biases[i] -= self.bias_costs[i] * learn_rate;
        }
    }

    fn calculate_hidden_layer_node_values(
        &mut self,
        layer: usize,
        weighted_inputs: [f64; num_nodes(SIZES)],
    ) {
        for node in 0..idx(SIZES, layer) {
            let node_idx = self.get_node_index(layer, node);
            let mut output_derivative = 0.;
            for output in 0..idx(SIZES, layer + 1) {
                output_derivative += self.weight_costs
                    [self.get_weight_index(layer + 1, output, node)]
                    * self.node_values[self.get_node_index(layer + 1, output)];
            }
            self.node_values[node_idx] =
                output_derivative * Activation::derivative(weighted_inputs[node_idx]);
        }
    }

    fn calculate_output_layer_node_values(
        &mut self,
        actual_outputs: [f64; num_nodes(SIZES)],
        expected_outputs: [f64; outputs(SIZES)],
        weighted_inputs: [f64; num_nodes(SIZES)],
    ) {
        for node in 0..(outputs(SIZES)) {
            let idx = self.get_node_index(SIZES.len() - 1, node);

            self.node_values[idx] = self
                .cost_derivative(actual_outputs[idx], expected_outputs[node])
                * Activation::derivative(weighted_inputs[idx]);
            // println!(
            //     "activation nv {} {}",
            //     actual_outputs[idx], expected_outputs[node]
            // );
        }
    }

    fn calculate_layer_weight_and_bias_costs(
        &mut self,
        layer: usize,
        actual_outputs: [f64; num_nodes(SIZES)],
    ) {
        for output in 0..idx(SIZES, layer) {
            let out_idx = self.get_node_index(layer, output);
            // Weights
            for input in 0..idx(SIZES, layer - 1) {
                self.weight_costs[self.get_weight_index(layer, output, input)] += actual_outputs
                    [self.get_node_index(layer - 1, input)]
                    * self.node_values[out_idx];
            }
            // Bias
            // println!("{}", self.node_values[out_idx]);
            self.bias_costs[out_idx] += self.node_values[out_idx];
        }
    }

    pub fn learn<const T: usize>(&mut self, data: [DataPoint<SIZES>; T], learn_rate: f64) {
        // Reset all node values and weight/bias cost
        self.bias_costs = [0.; num_nodes(SIZES)];
        self.weight_costs = [0.; num_weights(SIZES)];
        self.node_values = [0.; num_nodes(SIZES)];

        // Could make this multithreaded
        for data_point in data {
            let (actual_outputs, weighted_inputs) =
                self.calculate_all_with_weighted_inputs(data_point.inputs);
            // Go backwards (start with last one)
            self.calculate_output_layer_node_values(
                actual_outputs,
                data_point.expected,
                weighted_inputs,
            );
            self.calculate_layer_weight_and_bias_costs(SIZES.len() - 1, actual_outputs);

            for layer in (1..(SIZES.len() - 1)).rev() {
                self.calculate_hidden_layer_node_values(layer, weighted_inputs);
                self.calculate_layer_weight_and_bias_costs(layer, actual_outputs)
            }
        }

        self.update_weights_and_biases(learn_rate / T as f64);
    }
}
