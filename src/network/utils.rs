use std::f64::consts::PI;

pub fn arrFromFunc<T>(len: usize, func: fn(index: usize) -> T) -> Vec<T> {
    let mut a: Vec<T> = Vec::with_capacity(len);
    for i in 0..len {
        //(let i = 0; i < len; i++) {
        a.push(func(i));
        // console.log(a[i]);
    }

    a
}

pub fn randomInNormalDistribution(sd: f64, mean: f64) -> f64 {
    let x: f64 = rand::random();
    let y: f64 = rand::random();
    let z = (-2. * (x).ln()).sqrt() * (2. * PI * y).cos(); // Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * y);
    z * sd + mean
}

// export const flip = (bit: 0 | 1) => {
//     return Math.abs(bit - 1);
// };

pub fn maxIdx(arr: &Vec<f64>) -> usize {
    let mut max = arr[0];
    let mut maxIdx = 0;
    for i in 0..arr.len() {
        if arr[i] > max {
            max = arr[i];
            maxIdx = i;
        }
    }

    maxIdx
}

// export const maxIdx = (arr: Vec<f64>) => {
//     return arr.reduce(
//         (prev, num, i) => {
//             if (num > prev.max) {
//                 return { max: num, idx: i };
//             } else {
//                 return prev;
//             }
//         },
//         { max: -Infinity, idx: -1 }
//     ).idx;
// };
