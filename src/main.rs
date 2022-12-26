#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
pub mod network;

use std::f64::consts::PI;

use network::Network;
fn main() {
    let mut NN = Network::new(&[2, 4, 2, 2]);

    let data = generateData();
    println!("{:?}", data);
    print!("Starting...");
    for i in 0..300 {
        let (accuracy, cost) = NN.train(data.clone(), 1., true);
        print!("\rRUN {}:  Acc: {:.5}, Cost: {:.5}", i, accuracy, cost);
    }

    println!();
    println!("{:#?}", NN);
}

fn isValid(a: f64, b: f64) -> bool {
    (2. * PI * a).cos() / 2. + 0.4 > b
}

fn generateData() -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut data: Vec<(Vec<f64>, Vec<f64>)> = vec![];

    for _i in 0..1000 {
        let a = rand::random::<f64>();
        let b = rand::random::<f64>();

        let inputs = vec![a, b];
        if (isValid(a, b)) {
            data.push((inputs, vec![1., 0.]));
        } else {
            data.push((inputs, vec![0., 1.]));
        }
    }

    data
}
