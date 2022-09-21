pub struct Activation {}
pub struct Cost {}

/******************** CHANGE THINGS IN HERE ********************/

impl Activation {
    pub fn function(weighted_sum: f64) -> f64 {
        if weighted_sum > 0. {
            weighted_sum
        } else {
            0.
        }
    }

    pub fn derivative(weighted_sum: f64) -> f64 {
        if weighted_sum > 0. {
            1.
        } else {
            0.
        }
    }

    pub fn output_function<const T: usize>(weighted_sums: &[f64; T], idx: usize) -> f64 {
        let mut exp_sum = 0.;
        for weighted_sum in weighted_sums {
            exp_sum += weighted_sum.exp();
        }

        weighted_sums[idx].exp() / exp_sum
    }

    pub fn output_derivative<const T: usize>(weighted_sums: &[f64; T], idx: usize) -> f64 {
        let mut exp_sum = 0.;
        for weighted_sum in weighted_sums {
            exp_sum += weighted_sum.exp();
        }

        let ex = weighted_sums[idx].exp();

        (ex * exp_sum - ex * ex) / (exp_sum * exp_sum)
    }
}
