use super::{layer::Layer, Network};

pub struct DataPointRunData {
    pub activations: Vec<Vec<f64>>,
    pub weightedSums: Vec<Vec<f64>>,
    pub nodeValues: Vec<Vec<f64>>,
    pub inputs: Vec<f64>,
}

impl DataPointRunData {
    pub fn new(inputs: Vec<f64>, network: &Network) -> DataPointRunData {
        let mut layer = DataPointRunData {
            inputs,
            weightedSums: vec![],
            nodeValues: vec![],
            activations: vec![],
        };

        for i in 0..network.layerSizes.len() {
            //(let i = 0; i < network.layerSizes.length; i++) {
            layer.activations.push(vec![0.; network.layerSizes[i]]);
            layer.weightedSums.push(vec![0.; network.layerSizes[i]]);
            layer.nodeValues.push(vec![0.; network.layerSizes[i]]);

            // this.activations.push(new Array(network.layerSizes[i]).fill(0));
            // this.weightedSums.push(new Array(network.layerSizes[i]).fill(0));
            // this.nodeValues.push(new Array(network.layerSizes[i]).fill(0));
        }
        layer
    }
}
