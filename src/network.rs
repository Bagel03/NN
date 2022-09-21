use self::layer::Layer;

mod layer;
mod utils;
pub struct Network<const LAYER_SIZES: &'static [usize]> {
    pub layers: [Layer; Self::len()],

}

impl<const T: &'static [usize]> Network<T> {
    pub fn new() -> Network<T> {

        Network {}
    }

    fn create_layers() -> []
}
