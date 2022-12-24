use super::Activation;

pub fn get_ReLU() -> Activation {
    Activation {
        function: |inputs, into| {
            for i in 0..inputs.len() {
                into[i] = inputs[i].max(0.);
            }
        },
        derivative: |inputs| {
            let mut outputs = vec![0.; inputs.len()];
            for i in 0..inputs.len() {
                outputs[i] = if inputs[i] <= 0. { 0. } else { 1. };
            }
            outputs
        },
    }
}
