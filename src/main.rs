#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![feature(inherent_associated_types)]
#![feature(const_for)]
// #![feature(generic_arg_infer)]
#![feature(const_mut_refs)]

use crate::network::{cost::DataPoint, *};
mod activation;
mod network;

fn main() {
    const layer_sizes: &[usize] = &[2, 3, 2];
    let training_data: [DataPoint<{ &layer_sizes }>; 4] = [
        DataPoint::new([0., 1.], [1., 0.]),
        DataPoint::new([1., 0.], [1., 0.]),
        DataPoint::new([1., 1.], [1., 0.]),
        DataPoint::new([0., 0.], [0., 1.]),
    ];

    let mut network = Network::<{ &layer_sizes }>::new();

    for i in 0..5000 {
        network.learn([], 0.01);
    }

    // println!("{:#?}", &network);

    let res = network.calculate_outputs([0., 1.]);

    println!("{:#?}", &res);
}
