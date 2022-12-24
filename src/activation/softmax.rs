use super::Activation;

pub fn get_softmax() -> Activation {
    Activation {
        function: |inputs, into| {
            let mut expSum = 0.;
            for input in inputs {
                expSum += input.exp();
            }

            for i in 0..inputs.len() {
                into[i] = inputs[i].exp() / expSum
            }
        },
        derivative: |inputs| {
            let mut expSum = 0.;
            for input in inputs {
                expSum += input.exp();
            }

            let mut derivatives = vec![0.; inputs.len()];

            for i in 0..inputs.len() {
                let exp = inputs[i].exp();
                derivatives[i] = (exp * expSum - exp * exp) / (expSum * expSum);
            }

            derivatives
        },
    }
}
