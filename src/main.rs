mod super_matrix;
mod solver;
mod test_super_matrix;

extern crate superlu_sys;

use super_matrix::SuperMatrix;
use ndarray::arr1;
use std::os::raw::c_int;
use sprs::CsMat;
use superlu_sys::colperm_t::NATURAL;

fn main() {
    let nnz = 12;
    let m = 5;
    let n = 5;

    // Values of the non-zero elements
    let values = vec![19.0, 12.0, 12.0, 21.0, 12.0, 12.0, 21.0, 16.0, 21.0, 5.0, 21.0, 18.0];

    // Corresponding row indices
    let row_indices = vec![0, 1, 4, 1, 2, 4, 0, 2, 0, 3, 3, 4];

    // Column pointers
    let col_ptrs = vec![0, 3, 6, 8, 10, 12];

    let A = CsMat::new_csc((m, n), col_ptrs, row_indices, values);

    let B = vec![arr1(&[1., 1., 1., 1., 1.])];

    let options = solver::Options::default();

    let res = solver::solve(A, &B, &options);
    println!("{:?}", res);
    /*
    let mut L = SuperMatrix::default();
    let mut U = SuperMatrix::default();
    let mut options = superlu_options_t::default();
    let mut stat = superlu_sys::SuperLUStat_t::default();

    unsafe {
        let mut opt = solver::Options::default();
        opt.ffi.ColPerm = NATURAL;

        let mut perm_r = vec![0; m];
        let mut perm_c = vec![0; n];

        StatInit(&mut stat);

        let mut info = 0;
        dgssv(&mut options, &mut A, perm_c.as_mut_ptr(), perm_r.as_mut_ptr(), &mut L, &mut U, &mut B, &mut stat, &mut info);

        dPrint_CompCol_Matrix(b"A\0".as_ptr() as *const _, &A);
        dPrint_CompCol_Matrix(b"U\0".as_ptr() as *const _, &U);
        dPrint_SuperNode_Matrix(b"L\0".as_ptr() as *const _, &L);
        print_int_vec(b"\nperm_r\0".as_ptr() as *const _, m as c_int, perm_r.as_mut_ptr());

        SUPERLU_FREE(rhs.as_mut_ptr() as *mut c_void);
        SUPERLU_FREE(perm_r.as_mut_ptr() as *mut c_void);
        SUPERLU_FREE(perm_c.as_mut_ptr() as *mut c_void);
        Destroy_SuperNode_Matrix(&mut L);
        Destroy_CompCol_Matrix(&mut U);
        StatFree(&mut stat);
    }
     */
}
