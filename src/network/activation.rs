use std::fmt::Debug;

pub struct Activation {
    pub function: fn(weightedSums: &Vec<f64>, into: &mut Vec<f64>),
    pub derivative: fn(weightedSums: &Vec<f64>) -> Vec<f64>,
}

// abstract class SimpleActivation extends Activation {
//     abstract singleFunction(weightedSum: number) -> number;
//     abstract singleDerivative(weightedSum: number) -> number;

//     function(weightedSums: Vec<f64>, into: Vec<f64>) {
//         return weightedSums.map((n) => this.singleFunction(n));
//     }

//     derivative(weightedSums: Vec<f64>) -> Vec<f64> {
//         return weightedSums.map((n) => this.singleDerivative(n));
//     }
// }

impl Debug for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Activation").finish()
    }
}

pub const ReLU: Activation = Activation {
    function: |weightedSums, into| {
        for i in 0..weightedSums.len() {
            into[i] = weightedSums[i].max(0.); //Math.max(weightedSums[i], 0);
        }
    },

    derivative: |weightedSums| -> Vec<f64> {
        let mut result = vec![0.; weightedSums.len()]; // new Array(weightedSums.length).fill(0);
        for i in 0..weightedSums.len() {
            if weightedSums[i] > 0. {
                result[i] = 1.;
            }
        }
        return result;
    },
};

pub const Sigmoid: Activation = Activation {
    function: |weightedSums, into| {
        for i in 0..weightedSums.len() {
            into[i] = 1. / (1. + (-weightedSums[i]).exp())
        }
    },

    derivative: |weightedSums| -> Vec<f64> {
        let mut result = vec![0.; weightedSums.len()]; // new Array(weightedSums.length).fill(0);
        for i in 0..weightedSums.len() {
            let a = 1. / (1. + (-weightedSums[i]).exp());
            result[i] = a * (1. - a);
        }
        return result;
    },
};

pub const SoftMax: Activation = Activation {
    function: |weightedSums, into| {
        let mut sum = 0.;
        for i in 0..weightedSums.len() {
            sum += (weightedSums[i]).exp();
        }

        for i in 0..weightedSums.len() {
            into[i] = (weightedSums[i]).exp() / sum;
        }
    },

    derivative: |weightedSums| -> Vec<f64> {
        let mut result = vec![0.; weightedSums.len()]; // new Array(weightedSums.length).fill(0);

        let mut sum = 0.;
        for i in 0..weightedSums.len() {
            sum += (weightedSums[i]).exp();
        }

        for i in 0..weightedSums.len() {
            let exp = (weightedSums[i]).exp();
            result[i] = (exp * sum - exp * exp) / (sum * sum);
        }

        return result;
        // return weightedSums.map((n) => {
        //     const exp = Math.exp(n);
        //     return (exp * expSum - exp * exp) / (expSum * expSum);
        // });
    },
};
