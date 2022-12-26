// export abstract class Cost {
//     abstract function(
//         generatedResults: Vec<f64>,
//         correctResults: Vec<f64>
//     ) -> number;
//     abstract derivative(
//         generatedResults: Vec<f64>,
//         correctResults: Vec<f64>
//     ) -> Vec<f64>;

//     static MSE: Cost;
//     static CrossEntropy: Cost;
// }

pub struct Cost {}

impl Cost {
    pub fn function(generatedResults: &Vec<f64>, correctResults: &Vec<f64>) -> f64 {
        let mut sum = 0.;

        for i in 0..generatedResults.len() {
            // (let i = 0; i < generatedResults.length; i++) {
            let v = if (correctResults[i] == 1.) {
                generatedResults[i].ln()
                //Math.log(generatedResults[i])
            } else {
                (1. - generatedResults[i]).ln()
                // Math.log(1 - generatedResults[i])
            };

            if (!v.is_nan()) {
                sum += v
            }
        }

        return -sum;
    }

    // static function(generatedResults: Vec<f64>, correctResults: Vec<f64>) -> number {
    //     return -generatedResults.reduce((prev, generated, i) => {
    //         const correct = correctResults[i];

    //         const v =
    //             correct == 1 ? Math.log(generated) : Math.log(1 - generated);
    //         return prev + (Number.isNaN(v) ? 0 : v);
    //     }, 0);
    // }

    pub fn derivative(generatedResults: &Vec<f64>, correctResults: &Vec<f64>) -> Vec<f64> {
        let mut result = vec![0.; generatedResults.len()];

        for i in 0..generatedResults.len() {
            // (let i = 0; i < generatedResults.length; i++) {
            if (generatedResults[i] == 0. || generatedResults[i] == 1.) {
                continue;
            }
            let correct = correctResults[i];
            let generated = generatedResults[i];

            result[i] = (correct - generated) / (generated * (generated - 1.));
        }

        return result;
    }

    // static _derivative(generatedResults: Vec<f64>, correctResults: Vec<f64>) -> Vec<f64> {
    //     return generatedResults.map((generated, i) => {
    //         const correct = correctResults[i];

    //         // console.log(correct, generated)
    //         if (generated == 0 || generated == 1) {
    //             console.log("solved")
    //             return 0;

    //         }
    //         return (correct - generated) / (generated * (generated - 1));
    //     });
    // }
}
