use crate::layer::Layer;

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    inputs: usize,
}

impl Network {
    pub fn new<const T: usize>(layer_sizes: [usize; T]) -> Network {
        let mut layers = Vec::new();
        for layer in 1..layer_sizes.len() {
            layers.push(Layer::new(layer_sizes[layer - 1], layer_sizes[layer]));
        }

        Network {
            layers,
            inputs: layer_sizes[0],
        }
    }

    pub fn calculate_outputs<const T: usize>(&self, inputs: [f64; T]) -> Vec<f64> {
        assert_eq!(inputs.len(), self.inputs, "Wrong number of inputs");

        let mut current_outputs = Vec::from(inputs);
        for layer in self.layers.iter() {
            current_outputs = layer.calculate_outputs(&current_outputs);
        }

        current_outputs
    }

    pub fn total_cost<const T: usize>(&self, inputs: [f64; T], expected_outputs: [f64; T]) -> f64 {
        let outputs = self.calculate_outputs(inputs);
        outputs
            .iter()
            .enumerate()
            .map(|(i, num)| Layer::node_cost(num, &expected_outputs[i]))
            .reduce(|accum, num| accum + num)
            .unwrap()
    }
}
