use std::mem::MaybeUninit;
use sprs::{bmat, CsMat};
use ndarray::{arr2, Array1, Array2, Axis};
use superlu_sys::{superlu_options_t, yes_no_t};
use superlu_sys as ffi;
use superlu_sys::colperm_t::NATURAL;
use crate::solver::SolverError::Diverged;
use crate::SuperMatrix;

#[derive(Debug)]
pub enum SolverError {
    Conflict,
    Diverged,
    SetupError(Vec<[usize; 3]>),
    QuestionsOpen,
    UnknownError(String),
}

pub struct Options {
    pub ffi: superlu_options_t
}

impl Default for Options {
    fn default() -> Self {
        Self {
            ffi:
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
    }
}

fn vec_of_array1_to_array2(columns: &Vec<Array1<f64>>) -> Array2<f64> {
    let nrows = columns.first().map_or(0, |first_col| first_col.len());
    let ncols = columns.len();
    let mut result = Array2::zeros((nrows, ncols));

    for (col_idx, col) in columns.iter().enumerate() {
        result.column_mut(col_idx).assign(col);
    }

    result
}

pub fn solve (a: CsMat<f64>, b: &Vec<Array1<f64>>, options: &mut Options) -> Result<Vec<Array1<f64>>, SolverError> {
    use superlu_sys::Dtype_t::*;
    use superlu_sys::Mtype_t::*;
    use superlu_sys::Stype_t::*;

    let m = a.rows();
    let n = a.cols();
    if m != n {return Err(SolverError::Conflict)}
    if b.len() > 0 {
        if m != b[0].len() {return Err(SolverError::Conflict)}
        for rhs_col in b {
            if rhs_col.len() != b[0].len() {return Err(SolverError::Conflict)}
        }
    }


    let mut a_mat = SuperMatrix::from_csc_mat(a);
    let mut b_mat = SuperMatrix::from_ndarray(vec_of_array1_to_array2(b));

    let res_data = unsafe {

        let (m, n, nnz) = (5, 5, 12);

        let perm_r = ffi::intMalloc(m);
        assert!(!perm_r.is_null());

        let perm_c = ffi::intMalloc(n);
        assert!(!perm_c.is_null());

        ffi::set_default_options(&mut options.ffi);

        let mut stat: ffi::SuperLUStat_t = MaybeUninit::zeroed().assume_init();
        ffi::StatInit(&mut stat);

        let mut l_mat: ffi::SuperMatrix = MaybeUninit::zeroed().assume_init();
        let mut u_mat: ffi::SuperMatrix = MaybeUninit::zeroed().assume_init();

        let mut info = 0;
        ffi::dgssv(
            &mut options.ffi,
            a_mat.raw_mut(),
            perm_c,
            perm_r,
            &mut l_mat,
            &mut u_mat,
            b_mat.raw_mut(),
            &mut stat,
            &mut info,
        );
        if info != 0 {return Err(Diverged)}
        let res_data = b_mat.raw().data_as_vec();
        ffi::SUPERLU_FREE(perm_r as *mut _);
        ffi::SUPERLU_FREE(perm_c as *mut _);
        ffi::Destroy_SuperNode_Matrix(&mut l_mat);
        ffi::Destroy_CompCol_Matrix(&mut u_mat);
        ffi::StatFree(&mut stat);
        res_data
    };

    match res_data {
        None => {Err(Diverged)}
        Some(data) => {
            Ok(data
                .chunks(n)
                .map(|chunk| Array1::from_iter(chunk.iter().cloned()))
                .collect())
        }
    }

}
