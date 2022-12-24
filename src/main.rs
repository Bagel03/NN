mod activation;
mod network;

use activation::Activation;
use network::DataPoint;
use rand::random;

use crate::activation::{sigmoid::get_sigmoid, softmax::get_softmax};
// use network::Network;

fn main() {
    let mut model = network::Network::new(vec![2, 4, 2, 2], get_sigmoid(), get_softmax());

    println!("{:#?}", model.layers);

    let data = generate_data();

    print!("Starting...");

    let mut run = 0;
    for i in 0..1 {
        let run_data = model.train(&data, 1., 0.1, 0.9, true).unwrap();

        print!(
            "\rAccuracy: {}, Average Cost: {}  (Run {})",
            run_data.accuracy, run_data.average_cost, run
        );
        run += 1;
    }

    println!("\nDone...");

    println!("{:#?}", model.layers);
}

fn generate_data() -> Vec<DataPoint> {
    let mut v = Vec::with_capacity(1000);
    for i in 0..1000 {
        let a = random::<f64>();
        let b = random::<f64>();
        let result = (1.5 * a - 0.75).powi(3) + (0.01 * a).powi(2) + 0.5;
        let valid = ((result / b).floor() as f64).min(1.);

        v.push(DataPoint {
            inputs: vec![a, b],
            correct_outputs: vec![valid, -valid + 1.],
        })
    }
    v
}
