use super::sizes::*;

pub struct DataPoint<const SIZES: &'static [usize]>
where
    [(); idx(SIZES, 0)]:,
    [(); outputs(SIZES)]:,
{
    pub inputs: [f64; idx(SIZES, 0)],
    pub expected: [f64; outputs(SIZES)],
}

impl<const SIZES: &'static [usize]> super::Network<SIZES>
where
    [(); num_weights(SIZES)]:,
    [(); num_nodes(SIZES)]:,
    [(); SIZES.len()]:,
    [(); idx(SIZES, 0)]:,
    [(); outputs(SIZES)]:,
{
    pub fn cost(&self, data_point: DataPoint<SIZES>) -> f64 {
        let outputs = self.calculate_outputs(data_point.inputs);
        let mut cost = 0.;
        for i in 0..outputs.len() {
            let err = data_point.expected[i] - outputs[i];
            cost += err * err;
        }

        cost * 0.5
    }

    pub fn cost_derivative(&self, actual_output: f64, expected_output: f64) -> f64 {
        actual_output - expected_output
    }

    pub fn total_cost<const T: usize>(&self, data_points: [DataPoint<SIZES>; T]) -> f64 {
        let mut total_cost = 0.;
        for data_point in data_points {
            total_cost += self.cost(data_point);
        }
        total_cost / T as f64
    }
}