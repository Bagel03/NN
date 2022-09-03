// Cant use ranges, so no for loops :(

pub const fn num_weights(sizes: &[usize]) -> usize {
    let mut total = 0;
    let mut current_idx = 1;

    while current_idx < sizes.len() {
        total += sizes[current_idx] * sizes[current_idx - 1];
        current_idx += 1;
    }
    total
}

pub const fn num_nodes(sizes: &[usize]) -> usize {
    let mut total = 0;
    let mut current_idx = 0;

    while current_idx < sizes.len() {
        total += sizes[current_idx];
        current_idx += 1;
    }
    total
}

pub const fn idx(sizes: &[usize], idx: usize) -> usize {
    sizes[idx]
}

pub const fn outputs(sizes: &[usize]) -> usize {
    sizes[sizes.len() - 1]
}
