// Cant use ranges :(
const fn const_add_start_from_1(sizes: &[usize], mult_by_last: bool) -> usize {
    let mut total = 0;
    let mut current_idx = 1;

    while current_idx < sizes.len() {
        total += if mult_by_last {
            sizes[current_idx] * sizes[current_idx - 1]
        } else {
            sizes[current_idx]
        };
        current_idx += 1;
    }
    total
}

pub const fn num_weights(sizes: &[usize]) -> usize {
    const_add_start_from_1(sizes, true)
}

pub const fn num_nodes(sizes: &[usize]) -> usize {
    const_add_start_from_1(sizes, false)
}

pub const fn idx(sizes: &[usize], idx: usize) -> usize {
    sizes[idx]
}

pub const fn outputs(sizes: &[usize]) -> usize {
    idx(sizes, sizes.len() - 1)
}
