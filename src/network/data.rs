use super::Network;

#[derive(Debug)]
pub struct DataPointRunData {
    pub inputs: Vec<f64>,

    pub activations: Vec<Vec<f64>>,
    pub weighted_sums: Vec<Vec<f64>>,

    pub node_values: Vec<Vec<f64>>,
}

impl DataPointRunData {
    pub fn new(network: &Network, inputs: Vec<f64>) -> DataPointRunData {
        let mut data = DataPointRunData {
            inputs,
            activations: Vec::with_capacity(network.total_nodes),
            weighted_sums: Vec::with_capacity(network.total_nodes),
            node_values: Vec::with_capacity(network.total_nodes),
        };

        for i in 1..network.layer_sizes.len() {
            data.activations.push(vec![0.; network.layer_sizes[i]]);
            data.weighted_sums.push(vec![0.; network.layer_sizes[i]]);
            data.node_values.push(vec![0.; network.layer_sizes[i]]);
        }

        data
    }

    pub fn outputs(&self) -> &Vec<f64> {
        self.activations.last().unwrap()
    }
}
