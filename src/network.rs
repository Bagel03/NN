pub mod cost;
pub mod data;
pub mod layer;
pub mod utils;

use data::DataPointRunData;
use layer::Layer;

use crate::activation::Activation;

use self::{cost::Cost, utils::max_idx};

#[derive(Debug)]
pub struct DataPoint {
    pub inputs: Vec<f64>,
    pub correct_outputs: Vec<f64>,
}

pub struct LearnResults {
    pub accuracy: f64,
    pub average_cost: f64,
}

#[derive(Debug)]
pub struct Network {
    pub layer_sizes: Vec<usize>,
    pub layers: Vec<Layer>,
    pub total_nodes: usize,
}

impl Network {
    pub fn new(
        layer_sizes: Vec<usize>,
        activation: Activation,
        output_activation: Activation,
    ) -> Network {
        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);

        let mut total_nodes = 0;
        for i in 1..layer_sizes.len() {
            layers.push(Layer::new(
                i - 1,
                layer_sizes[i - 1],
                layer_sizes[i],
                activation,
            ));
            total_nodes += layer_sizes[i];
        }

        layers.last_mut().unwrap().activation = output_activation;

        Network {
            layers,
            layer_sizes,
            total_nodes,
        }
    }

    pub fn get_output_layer(&mut self) -> &mut Layer {
        self.layers.last_mut().unwrap()
    }

    pub fn feed_forward(&self, inputs: &Vec<f64>) -> DataPointRunData {
        let mut data = DataPointRunData::new(&self, inputs.to_owned());
        let data_ref = &mut data;

        for layer in self.layers.iter() {
            layer.feed_forward(data_ref);
        }

        data
    }

    pub fn train(
        &mut self,
        data_points: &Vec<DataPoint>,
        learn_rate: f64,
        regularization: f64,
        momentum: f64,
        store_results: bool,
    ) -> Option<LearnResults> {
        // Have to keep this
        let len = data_points.len();
        let mut results = LearnResults {
            accuracy: 0.,
            average_cost: 0.,
        };

        for data_point in data_points {
            let mut network_data = self.feed_forward(&data_point.inputs);

            self.get_output_layer()
                .calc_output_node_values(&mut network_data, &data_point.correct_outputs);
            self.get_output_layer().calc_gradients(&network_data);

            for i in (0..self.layers.len() - 2).rev() {
                self.layers[i]
                    .calc_hidden_layer_node_values(&mut network_data, &self.layers[i + 1]);
                self.layers[i].calc_gradients(&network_data)
            }

            if !store_results {
                continue;
            }

            let outputs = &network_data.activations[network_data.activations.len() - 1];
            if max_idx(&data_point.correct_outputs) == max_idx(outputs) {
                results.accuracy += 1.;
            }

            results.average_cost += Cost::function(&outputs, &data_point.correct_outputs)
        }

        for layer in self.layers.iter_mut() {
            layer.apply_gradients(learn_rate / len as f64, regularization, momentum);
        }

        if store_results {
            results.accuracy /= len as f64;
            results.average_cost /= len as f64;
            return Some(results);
        }

        None
    }
}
