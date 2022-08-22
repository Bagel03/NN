use crate::network::Network;

mod activation;
mod layer;
mod network;
fn main() {
    let network = Network::new([2, 3, 2]);
    // println!("{:#?}", &network);

    let res = network.calculate_outputs([0., 1.]);

    println!("{:#?}", &res);
}
