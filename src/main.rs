mod super_matrix;
mod test_super_matrix;


extern crate superlu_sys;

use super_matrix::SuperMatrix;
use superlu_sys::{superlu_options_t, yes_no_t};
use ndarray::arr2;
use std::os::raw::c_int;
use sprs::CsMat;

fn default_options () -> superlu_options_t {
    superlu_options_t {
        Fact: superlu_sys::fact_t::DOFACT,
        Equil: yes_no_t::YES,
        ColPerm: superlu_sys::colperm_t::COLAMD,
        Trans: superlu_sys::trans_t::NOTRANS,
        IterRefine: superlu_sys::IterRefine_t::NOREFINE,
        DiagPivotThresh: 1.0,
        SymmetricMode: yes_no_t::NO,
        PivotGrowth: yes_no_t::NO,
        ConditionNumber: yes_no_t::NO,
        PrintStat: yes_no_t::YES,
        RowPerm: superlu_sys::rowperm_t::LargeDiag,
        ILU_DropRule: 0,
        ILU_DropTol: 0.0,
        ILU_FillFactor: 0.0,
        ILU_Norm: superlu_sys::norm_t::ONE_NORM,
        ILU_FillTol: 0.0,
        ILU_MILU: superlu_sys::milu_t::SILU,
        ILU_MILU_Dim: 0.0,
        //The following fields are probably unused
        ParSymbFact: yes_no_t::NO,
        ReplaceTinyPivot: yes_no_t::NO,
        SolveInitialized: yes_no_t::NO,
        RefineInitialized: yes_no_t::NO,
        nnzL: 0,
        nnzU: 0,
        num_lookaheads: 0,
        lookahead_etree: yes_no_t::NO,
        SymPattern: yes_no_t::NO,
    }
}

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

    let A = SuperMatrix::from_csc_mat(CsMat::new_csc((m, n), col_ptrs, row_indices, values));

    let B = SuperMatrix::from_ndarray(arr2(&[[1., 1., 1., 1., 1.]]));

    /*
    let mut L = SuperMatrix::default();
    let mut U = SuperMatrix::default();
    let mut options = superlu_options_t::default();
    let mut stat = superlu_sys::SuperLUStat_t::default();

    unsafe {

        let mut perm_r = vec![0; m];
        let mut perm_c = vec![0; n];

        set_default_options(&mut options);
        options.ColPerm = NATURAL;

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
