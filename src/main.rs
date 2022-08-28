#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![feature(const_for)]
// #![feature(generic_arg_infer)]
#![feature(const_mut_refs)]

use crate::network::*;

mod activation;
mod network;

fn main() {
    let network = Network::<{ &[2, 3, 3, 2] }>::new();

    // println!("{:#?}", &network);

    let res = network.calculate_outputs([0., 1.]);

    println!("{:#?}", &res);
}
