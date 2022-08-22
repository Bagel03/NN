use crate::activation::Activation;
use std::usize;

// NODES&: number of nodes in this one,
// INPUTS: number of&& nodes in the one before
#[derive(Debug)]
pub struct Layer {
    inputs: usize,
    nodes: usize,
    weights: Vec<f64>,
    biases: Vec<f64>,
}

// pub trait Calculable {
//     fn calculate_outputs(&self, inputs: Vec<f64>);
// }

impl Layer {
    pub fn new(inputs: usize, nodes: usize) -> Layer {
        Layer {
            inputs,
            nodes,
            weights: vec![0.; inputs * nodes],
            biases: vec![0.; nodes],
        }
    }

    pub fn calculate_outputs(&self, inputs: &Vec<f64>) -> Vec<f64> {
        assert_eq!(
            self.inputs,
            inputs.len(),
            "Layer did not receive the expected number of inputs. Expected {} values, got {}.",
            self.inputs,
            inputs.len()
        );

        let mut outputs = vec![0.; self.nodes];

        for node in 0..self.nodes {
            let mut output = self.biases[node];

            for input in 0..self.inputs {
                output += inputs[input] * self.weights[node * self.inputs + input];
            }

            outputs[node] = Activation::function(output)
        }

        outputs
    }

    pub fn node_cost(actual: &f64, expected: &f64) -> f64 {
        let error = actual - expected;
        error * error
    }

    // Derivative of the node cost function
    pub fn node_cost_der(actual: &f64, expected: &f64) -> f64 {
        2. * (actual - expected)
    }
}
