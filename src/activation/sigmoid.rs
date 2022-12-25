use super::Activation;

pub fn get_sigmoid() -> Activation {
    Activation {
        function: |inputs, into| {
            for i in 0..inputs.len() {
                into[i] = 1. / (1. + (-inputs[i]).exp());
            }
        },
        derivative: |inputs| {
            let mut outputs = vec![0.; inputs.len()];
            for i in 0..inputs.len() {
                let a = 1. / (1. + (-inputs[i]).exp());
                outputs[i] = a * (1. - a);
            }
            outputs
        },
    }
}
