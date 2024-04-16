# SuperLU

A Rust interface for SuperLU, utilizing ndarray and sprs for matrix operations.

## Example Usage

```rust
use std::time::Duration;
use ndarray::arr1;
use sprs::CsMat;
use crate::{Options, solve_super_lu};

fn main() {
    let values = vec![
        19.0, 12.0, 12.0, 21.0, 12.0, 12.0, 21.0, 16.0, 21.0, 5.0, 21.0, 18.0,
    ];
    let row_indices = vec![0, 1, 4, 1, 2, 4, 0, 2, 0, 3, 3, 4];
    let col_ptrs = vec![0, 3, 6, 8, 10, 12];
    let a_mat = CsMat::new_csc((5, 5), col_ptrs, row_indices, values);
    let rhs_1 = arr1(&[1., 1., 1., 1., 1.]);
    let rhs_2 = arr1(&[2., 2., 2., 2., 2.]);
    let b_mat = vec![rhs_1, rhs_2];
    let mut options = Options::default();
    let res = solve_super_lu(a_mat, &b_mat, Some(Duration::from_secs(60)), &mut options);
    let x_1 = res[0];
    let x_2 = res[1];
    println!("A x_1 = rhs_1 -> x_1 =:\n{:?}", x_1);
    println!("A x_2 = rhs_2 -> x_2 =:\n{:?}", x_2);
}
