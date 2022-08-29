use super::sizes::*;
use crate::activation::Activation;

impl<const SIZES: &'static [usize]> super::Network<SIZES>
where
    [(); num_weights(SIZES)]:,
    [(); num_nodes(SIZES)]:,
    [(); SIZES.len()]:,
{
    fn update_weights_and_biases(&mut self, learn_rate: f64) {
        for i in 0..self.weights.len() {
            self.weights[i] -= self.weight_costs[i] * learn_rate;
            if i < self.biases.len() {
                self.biases[i] -= self.bias_costs[i] * learn_rate;
            }
        }
    }

    pub fn learn(&mut self, learn_rate: f64) {
        // let cost = self.total_cost();
        // Calculate the derivate for each node

        //Output layer needs the cost derivative added in

        self.update_weights_and_biases(learn_rate);
    }
}
