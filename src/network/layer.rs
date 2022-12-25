use std::f64::consts::PI;

use super::{cost::Cost, data::DataPointRunData};
use crate::activation::Activation;
use rand::random;

#[derive(Debug)]
pub struct Layer {
    pub idx: usize,

    pub inputs: usize,
    pub nodes: usize,

    pub weights: Vec<f64>,
    pub biases: Vec<f64>,

    pub weight_gradients: Vec<f64>,
    pub bias_gradients: Vec<f64>,

    pub weight_momentum: Vec<f64>,
    pub bias_momentum: Vec<f64>,

    pub activation: Activation,
}

impl Layer {
    pub fn new(idx: usize, inputs: usize, nodes: usize, activation: Activation) -> Layer {
        let mut layer = Layer {
            idx,
            inputs,
            nodes,
            weights: vec![0.; inputs * nodes],
            biases: vec![0.; nodes],
            weight_gradients: vec![0.; inputs * nodes],
            bias_gradients: vec![0.; nodes],
            weight_momentum: vec![0.; inputs * nodes],
            bias_momentum: vec![0.; nodes],
            activation,
        };

        // Setup weights
        for i in 0..nodes * inputs {
            let x = random::<f64>();
            let y = random::<f64>();
            layer.weights[i] =
                ((-2. * x.ln()).sqrt() * (2. * PI * y).cos()) / (inputs as f64).sqrt();
        }

        for i in 0..nodes {
            layer.biases[i] = random::<f64>() - 0.5;
        }

        layer
    }

    fn get_weight_idx(&self, node: usize, input: usize) -> usize {
        input * self.nodes + node
    }

    fn get_weight(&self, node: usize, input: usize) -> f64 {
        self.weights[self.get_weight_idx(node, input)]
    }

    pub fn feed_forward(&self, data: &mut DataPointRunData) {
        let inputs = if self.idx == 0 {
            &data.inputs
        } else {
            &data.activations[self.idx - 1]
        };

        let weighted_sums = &mut data.weighted_sums[self.idx];

        for i in 0..self.nodes {
            weighted_sums[i] = self.biases[i];
            for j in 0..self.inputs {
                weighted_sums[i] += inputs[j] * self.get_weight(i, j);
            }
        }

        // Do activations
        (self.activation.function)(weighted_sums, &mut data.activations[self.idx]);
    }

    pub fn calc_output_node_values(
        &mut self,
        data: &mut DataPointRunData,
        correct_results: &Vec<f64>,
    ) {
        let cost_derivatives = Cost::derivative(&data.activations[self.idx], &correct_results);

        let activation_derivatives = (self.activation.derivative)(&data.weighted_sums[self.idx]);

        for i in 0..self.nodes {
            data.node_values[self.idx][i] = cost_derivatives[i] * activation_derivatives[i];
        }
    }

    pub fn calc_hidden_layer_node_values(&self, data: &mut DataPointRunData, next_layer: &Layer) {
        let activation_derivatives = (self.activation.derivative)(&data.weighted_sums[self.idx]);

        for node in 0..self.nodes {
            let mut sum = 0.;
            for output in 0..data.node_values[self.idx + 1].len() {
                sum += next_layer.get_weight(output, node) * data.node_values[self.idx + 1][output];
            }

            data.node_values[self.idx][node] = sum * activation_derivatives[node];
        }
    }

    pub fn calc_gradients(&mut self, data: &DataPointRunData) {
        let inputs = if self.idx == 0 {
            &data.inputs
        } else {
            &data.activations[self.idx - 1]
        };

        for node in 0..self.nodes {
            for input in 0..inputs.len() {
                self.weight_gradients[input * self.nodes + node] +=
                    data.node_values[self.idx][node] * inputs[input]
            }
            self.bias_gradients[node] += data.node_values[self.idx][node];
        }
    }

    pub fn apply_gradients(&mut self, learn_rate: f64, regularization: f64, momentum: f64) {
        let weight_decay = 1. - regularization * learn_rate;

        for i in 0..(self.nodes * self.inputs) {
            self.weight_momentum[i] =
                self.weight_momentum[i] * momentum - self.weight_gradients[i] * learn_rate;

            self.weights[i] = self.weights[i] * weight_decay + self.weight_momentum[i];
            self.weight_gradients[i] = 0.;
        }

        for j in 0..self.nodes {
            self.bias_momentum[j] =
                self.bias_momentum[j] * momentum - self.bias_gradients[j] * learn_rate;

            self.biases[j] += self.bias_gradients[j];
            self.bias_gradients[j] = 0.;
        }
    }
}
