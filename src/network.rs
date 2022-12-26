// import { Activation } from "./network/activation.js";
// import { Cost } from "./network/cost.js";
// import { DataPointRunData } from "./network/data.js";
// import { Layer } from "./network/layer.js";
// import { maxIdx } from "./network/utils.js";

use self::cost::Cost;
use self::utils::*;
use self::{data::DataPointRunData, layer::Layer};

pub mod activation;
pub mod cost;
pub mod data;
pub mod layer;
pub mod utils;
#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    layerSizes: Vec<usize>,
    num_layers: usize,
}

impl Network {
    pub fn new(layerSizes: &[usize]) -> Network {
        let mut network = Network {
            layers: vec![],
            layerSizes: layerSizes[1..].to_vec(),
            num_layers: layerSizes.len() - 1,
        };
        for i in 1..layerSizes.len() {
            // (let i = 1; i < layerSizes.length; i++) {
            let activation = if i == layerSizes.len() - 1 {
                activation::SoftMax
            } else {
                activation::Sigmoid
            };
            network.layers.push(Layer::new(
                layerSizes[i],
                layerSizes[i - 1],
                activation,
                i - 1,
            ))
        }

        network
        // self.layers = hiddenLayerSizes.map(
        //     (nodes, i) =>
        //         new Layer(
        //             nodes,
        //             i == 0 ? inputs : hiddenLayerSizes[i - 1],
        //             i == hiddenLayerSizes.length - 1
        //                 ? Activation.SoftMax
        //                 : Activation.Sigmod,
        //             i
        //         )
        // );
    }

    pub fn feedForward(&self, inputs: Vec<f64>) -> DataPointRunData {
        let mut data = DataPointRunData::new(inputs, &self);
        for layer in self.layers.iter() {
            //(const layer of self.layers) {
            layer.feedForward(&mut data);
        }
        return data;
    }

    // trains, returns if guess was correct
    // pub fn trainOnSinglePoint(&mut self,inputs: Vec<f64>, correctOutputs: Vec<f64>) -> boolean {
    //     let data = self.feedForward(inputs)
    //     const guess = maxIdx(data.activations[data.activations.length - 1]);

    //     self.layers[self.layers.length - 1].calcOutputNodeValues(data, correctOutputs);
    //     self.layers[self.layers.length - 1].calcGradients(data);

    //     for i in 0..self(let i = self.layers.length - 2; i >= 0; i--) {
    //         self.layers[i].calcHiddenLayerNodeValues(data, self.layers[i + 1]);
    //         self.layers[i].calcGradients(data);
    //     }

    //     return guess == maxIdx(correctOutputs);
    // }

    // (inputs, correctOutputs)[]
    // (accuracy, cost)
    pub fn train(
        &mut self,
        dataPoints: Vec<(Vec<f64>, Vec<f64>)>,
        learnRate: f64,
        storeData: bool,
    ) -> (f64, f64) {
        let mut correct = 0;
        let mut totalCost = 0.;

        let num_points = dataPoints.len();

        for dataPoint in dataPoints {
            //(const [inputs, correctOutputs] of dataPoints) {
            let (inputs, correctOutputs) = dataPoint;
            // println!("Correct: {:?}", correctOutputs);

            let mut data = self.feedForward(inputs);

            self.layers[self.num_layers - 1].calcOutputNodeValues(&mut data, &correctOutputs);
            self.layers[self.num_layers - 1].calcGradients(&mut data);

            for i in (0..self.layers.len() - 2).rev() {
                //(let i = self.layers.length - 2; i >= 0; i--) {
                self.layers[i].calcHiddenLayerNodeValues(&mut data, &self.layers[i + 1]);
                self.layers[i].calcGradients(&mut data);
            }

            if !storeData {
                continue;
            }

            let guess = maxIdx(&data.activations[data.activations.len() - 1]);
            if guess == maxIdx(&correctOutputs) {
                correct += 1;
            }

            // println!("{:?}", data.weightedSums[data.weightedSums.len() - 1]);
            // println!("{:?} ", data.activations[data.activations.len() - 1]);

            totalCost += Cost::function(
                &data.activations[data.activations.len() - 1],
                &correctOutputs,
            );

            // if (self.trainOnSinglePoint(inputs, correctOutputs)) correct++;
        }

        for layer in self.layers.iter_mut() {
            // (const layer of self.layers) {
            layer.applyGradients(learnRate / num_points as f64, 0.1, 0.9)
        }

        return (
            correct as f64 / num_points as f64,
            totalCost / num_points as f64,
        );
    }

    // totalCost(dataPoints: [inputs: Vec<f64>, correctOutputs: Vec<f64>][]) {
    //     const results = dataPoints.map((point) => self.feedForward(point[0]));

    //     const totalCost = results.reduce(
    //         (prev, curr, i) =>
    //             prev + Cost.function(curr.activations[curr.activations.length - 1], dataPoints[i][1]),
    //         0
    //     );
    //     return totalCost / dataPoints.length;
    // }
}
