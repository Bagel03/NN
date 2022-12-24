pub struct Cost {}

impl Cost {
    pub fn function(generated_results: &Vec<f64>, correct_results: &Vec<f64>) -> f64 {
        let mut sum = 0.;

        for i in 0..generated_results.len() {
            let v = if correct_results[i] == 1. {
                generated_results[i].ln()
            } else {
                (1. - generated_results[i]).ln()
            };

            if !v.is_nan() {
                sum += v;
            }
        }

        -sum
    }
    pub fn derivative(generated_results: &Vec<f64>, correct_results: &Vec<f64>) -> Vec<f64> {
        let mut derivatives = vec![0.; generated_results.len()];

        for i in 0..generated_results.len() {
            let correct = correct_results[i];
            let generated = generated_results[i];

            if generated == 0. || generated == 1. {
                derivatives[i] = 0.;
            } else {
                derivatives[i] = (correct - generated) / (generated * (generated - 1.));
            }
        }

        derivatives
    }
}
