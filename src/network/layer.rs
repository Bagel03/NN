#[derive(PartialEq)]
pub enum LayerKind {
    Input,
    Hidden,
    Output,
}
pub struct Layer {
    num_nodes: usize,
    num_weights: usize,
    num_inputs: usize,

    node_offset: usize,
    weight_offset: usize,
    prev_layer_node_offset: usize,

    kind: LayerKind,
}

impl Layer {
    pub fn feed_with_weighted_sums<const NODES: usize, const WEIGHTS: usize>(
        &self,
        weights: &[f64; WEIGHTS],
        biases: &[f64; WEIGHTS],
        activations: &mut [f64; NODES],
        weighted_sums: &mut [f64; NODES],
    ) {
        if self.kind == LayerKind::Input {
            println!("WARNING: Input layer tried to feed forward, please skip");
            return;
        }

        // Loop through each one of this layers weights
        for weight in 0..self.num_weights {
            let weight_idx = weight + self.weight_offset;

            let node = weight / self.num_inputs;
            let input = weight % self.num_inputs;

            weighted_sums[self.node_offset + node] +=
                weights[weight_idx] * activations[self.prev_layer_node_offset + input];
        }
    }
}
