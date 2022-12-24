pub fn max_idx(vec: &Vec<f64>) -> usize {
    let mut max = vec[0];
    let mut max_idx = 0;

    for i in 1..vec.len() {
        if max > vec[i] {
            max = vec[i];
            max_idx = i;
        }
    }

    max_idx
}
