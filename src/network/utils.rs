use super::Network;

impl<const T: &'static [usize]> Network<T> {
    pub const fn len() -> usize {
        T.len()
    }

    pub const fn inputs() -> usize {
        T[0]
    }

    pub const fn outputs() -> usize {
        T[T.len() - 1]
    }

    // Totals
    pub const fn total_nodes() -> usize {
        let (mut i, mut total) = (0, 0);
        while i < T.len() {
            total += T[i];
            i += 1;
        }
        total
    }

    pub const fn total_weights() -> usize {
        let (mut i, mut total) = (1, 0);
        while i < T.len() {
            total += T[i] * T[i - 1];
            i += 1;
        }
        total
    }

    // Offsets
    pub const fn node_offset(layer: usize) -> usize {
        let (mut i, mut total) = (0, 0);
        while i < layer {
            total += T[i];
            i += 1;
        }
        total
    }

    pub const fn weight_offset(layer: usize) -> usize {
        if layer == 0 {
            panic!("WARNING: first layer does not have any weights");
        }
        let (mut i, mut total) = (1, 0);
        while i < layer {
            total += T[i] * T[i - 1];
            i += 1
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use crate::network::Network;
    type Nw = Network<{ &[2, 4, 3, 2] }>;

    #[test]
    fn total_nodes() {
        let res = Nw::total_nodes();
        assert_eq!(res, 11);
    }

    #[test]
    fn total_weights() {
        let res = Nw::total_weights();
        assert_eq!(res, 26);
    }
    #[test]

    fn node_offsets() {
        let res = Nw::node_offset(2);
        assert_eq!(res, 6)
    }
    #[test]

    fn weight_offsets() {
        let res = Nw::weight_offset(2);
        assert_eq!(res, 8);
    }
}
