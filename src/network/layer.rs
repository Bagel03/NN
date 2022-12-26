use super::{
    activation::Activation,
    cost::Cost,
    data::DataPointRunData,
    utils::{arrFromFunc, randomInNormalDistribution},
};
#[derive(Debug)]

pub struct Layer {
    nodes: usize,
    inputs: usize,

    idx: usize,
    activation: Activation,

    weights: Vec<f64>,
    biases: Vec<f64>,

    weightGradients: Vec<f64>,
    biasGradients: Vec<f64>,

    weightVelocities: Vec<f64>,
    biasVelocities: Vec<f64>,
}

impl Layer {
    pub fn new(nodes: usize, inputs: usize, activation: Activation, idx: usize) -> Layer {
        let mut layer = Layer {
            nodes,
            inputs,
            idx,
            activation,
            weights: vec![],
            biases: vec![],

            weightGradients: vec![0.; nodes * inputs],
            biasGradients: vec![0.; nodes],
            biasVelocities: vec![0.; nodes],
            weightVelocities: vec![0.; nodes * inputs],
        };

        layer.biases = arrFromFunc(nodes, |_| rand::random::<f64>() - 0.5);

        let sqrt = (inputs as f64).sqrt();
        for i in 0..nodes * inputs {
            let y = randomInNormalDistribution(1.0, 0.0);
            layer.weights.push(y / sqrt);
            //(let i = 0; i < len; i++) {
            // console.log(a[i]);
        }

        layer

        // self.weightGradients = new Array(nodes * inputs).fill(0);
        // self.biasGradients = new Array(nodes).fill(0);
        // self.biasVelocities = new Array(nodes).fill(0);
        // self.weightVelocities = new Array(nodes * inputs).fill(0);
    }

    fn getWeightIdx(&self, node: usize, input: usize) -> usize {
        return node * self.inputs + input;
    }

    fn getWeight(&self, node: usize, input: usize) -> f64 {
        return self.weights[self.getWeightIdx(node, input)];
    }

    pub fn feedForward(&self, data: &mut DataPointRunData) {
        let inputs = if (self.idx == 0) {
            &data.inputs
        } else {
            &data.activations[self.idx - 1]
        };

        for node in 0..self.nodes {
            // (let node = 0; node < self.nodes; node++) {
            let mut sum = self.biases[node];

            for input in 0..self.inputs {
                // (let input = 0; input < self.inputs; input++) {
                sum += inputs[input] * self.getWeight(node, input);
            }

            data.weightedSums[self.idx][node] = sum;
        }

        (self.activation.function)(
            &data.weightedSums[self.idx],
            &mut data.activations[self.idx],
        );
    }

    pub fn calcOutputNodeValues(&self, data: &mut DataPointRunData, correctResults: &Vec<f64>) {
        let costDerivatives = Cost::derivative(&data.activations[self.idx], correctResults);
        let activationDerivatives = (self.activation.derivative)(&data.weightedSums[self.idx]);

        for i in 0..self.nodes {
            //(let i = 0; i < self.nodes; i++) {
            data.nodeValues[self.idx][i] = costDerivatives[i] * activationDerivatives[i];
        }

        // data.nodeValues[self.idx] = costDerivatives.map(
        //     (c, i) => c * activationDerivatives[i]
        // );
    }

    pub fn calcHiddenLayerNodeValues(&self, data: &mut DataPointRunData, nextLayer: &Layer) {
        let activationDerivatives = (self.activation.derivative)(&data.weightedSums[self.idx]);

        for node in 0..self.nodes {
            // (let node = 0; node < self.nodes; node++) {
            let mut sum = 0.;
            for output in 0..nextLayer.nodes {
                //(let output = 0; output < nextLayer.nodes; output++) {
                sum += nextLayer.getWeight(output, node) * data.nodeValues[self.idx + 1][output]
            }

            data.nodeValues[self.idx][node] = activationDerivatives[node] * sum;
        }

        // data.nodeValues[self.idx] = activationDerivatives.map(
        //     (activationDerivative, node) => {
        //         return (
        //             activationDerivative *
        //             data.nodeValues[self.idx + 1].reduce(
        //                 (prev, outputNodeValue, output) =>
        //                     prev +
        //                     nextLayer.getWeight(output, node) * outputNodeValue, 0
        //             )
        //         );
        //     }
        // );
    }

    pub fn calcGradients(&mut self, data: &DataPointRunData) {
        let inputs = if (self.idx == 0) {
            &data.inputs
        } else {
            &data.activations[self.idx - 1]
        };

        for node in 0..self.nodes {
            //(let node = 0; node < self.nodes; node++) {
            for input in 0..self.inputs {
                // (let input = 0; input < self.inputs; input++) {
                self.weightGradients[node * self.inputs + input /*self.getWeightIdx(node, input)*/] +=
                    data.nodeValues[self.idx][node] * inputs[input];
            }

            self.biasGradients[node] += data.nodeValues[self.idx][node];
        }
        // data.nodeValues[self.idx].forEach((nodeValue, node) => {
        //     inputs.forEach((inputValue, input) => {
        //         self.weightGradients[self.getWeightIdx(node, input)] +=
        //             nodeValue * inputValue;
        //     });
        //     self.biasGradients[node] += nodeValue;
        // });
    }

    pub fn applyGradients(&mut self, learnRate: f64, regularization: f64, momentum: f64) {
        let weightDecay = 1. - regularization * learnRate;

        for i in 0..(self.nodes * self.inputs) {
            //(let i = 0; i < self.nodes * self.inputs; i++) {

            self.weightVelocities[i] =
                self.weightVelocities[i] * momentum - self.weightGradients[i] * learnRate;
            self.weights[i] = self.weights[i] * weightDecay + self.weightVelocities[i];
            self.weightGradients[i] = 0.;
        }

        // self.weightGradients.forEach((weightCost, i) => {
        //     self.weightVelocities[i] =
        //         self.weightVelocities[i] * momentum -
        //         self.weightGradients[i] * learnRate;
        //     self.weights[i] =
        //         self.weights[i] * weightDecay + self.weightVelocities[i];
        //     self.weightGradients[i] = 0;
        // });

        for i in 0..self.nodes {
            // (let i = 0; i < self.nodes; i++) {

            self.biasVelocities[i] =
                self.biasVelocities[i] * momentum - self.biasGradients[i] * learnRate;
            self.biases[i] += self.biasVelocities[i];

            self.biasGradients[i] = 0.;
        }

        // self.biasGradients.forEach((biasCost, i) => {
        //     self.biasVelocities[i] =
        //         self.biasVelocities[i] * momentum -
        //         self.biasGradients[i] * learnRate;
        //     self.biases[i] += self.biasVelocities[i];

        //     self.biasGradients[i] = 0;
        // });
    }
}
