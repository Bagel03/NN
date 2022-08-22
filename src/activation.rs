pub struct Activation {}

// CHANGE THIS
impl Activation {
    // Sigmoid
    pub fn function(value: f64) -> f64 {
        1. / (1. + std::f64::consts::E.powf(-value))
    }

    pub fn derivative(value: f64) -> f64 {
        let activation = Activation::function(value);
        activation * (1. - activation)
    }
}
