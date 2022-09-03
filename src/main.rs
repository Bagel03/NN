#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![feature(inherent_associated_types)]
#![feature(const_for)]
// #![feature(generic_arg_infer)]
#![feature(const_mut_refs)]
#![allow(incomplete_features)]

use rand::Rng;

use crate::network::{cost::DataPoint, *};
mod activation;
mod network;

const LAYER_SIZES: &[usize] = &[2, 3, 2];

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let mut rng = rand::thread_rng();

    let mut training_data: [DataPoint<{ &LAYER_SIZES }>; 100] =
        [DataPoint::new([0., 0.], [0., 0.]); 100];
    for i in 0..training_data.len() {
        let x = rng.gen();
        let y = rng.gen();
        let is_valid = if (x * y) < 0.25 { 1. } else { 0. };
        training_data[i] = DataPoint::new([x, y], [is_valid, -is_valid + 1.]);
    }
    // let mut training_data: [DataPoint<{ &LAYER_SIZES }>; 100] = [DataPoint::new([0.], [0.]); 100];
    // for i in 0..training_data.len() {
    //     let x = (rng.gen::<f64>() - 0.1) / 2.;
    //     training_data[i] = DataPoint::new([x], [(x + 0.1) * 2.]);
    // }

    println!("POINT: {:#?}", training_data[0]);

    let mut network = Network::<{ &LAYER_SIZES }>::new();
    println!(
        "Start: {:?}",
        network.calculate_outputs(training_data[0].inputs)
    );
    network.learn(training_data, 0.01);

    for i in 0..100000 {
        network.learn([training_data[0]], 10.);
        if (i % 10000 == 0) {
            print!(
                "COST({}): {:.5}\t",
                i / 10000,
                network.total_cost(training_data)
            );
        }
    }

    println!(
        "End: {:?}",
        network.calculate_outputs(training_data[0].inputs)
    );

    // println!("{:#?}", &network);

    let original_cost = network.total_cost(training_data);
    return;
    // loop {
    //     network.learn(training_data, 0.1);

    //     let cost = network.total_cost(training_data);

    //     println!("{:.5}:  {}", original_cost, cost);
    //     i += 1;

    //     // if i > 100 {
    //     //     break;
    // }
    // // }

    // println!("{:#?}", &network);

    // let res = (
    //     network.calculate_all_with_weighted_inputs([1., 0.]),
    //     network.calculate_outputs([1., 0.]),
    //     // network.calculate_outputs([0., 1.]),
    //     // network.calculate_outputs([1., 1.]),
    // );
    // println!("{:#?}, {:#?}", &res.0, res.1);
    // println!("{:#?}", &network.total_cost(training_data))
}
