use std::mem::MaybeUninit;
use libc::c_int;
use sprs::CsMat;
use ndarray::{Array1, Array2};
use superlu_sys::{superlu_options_t, yes_no_t};
use superlu_sys as ffi;
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
        let mut options: superlu_options_t = unsafe {MaybeUninit::zeroed().assume_init()};
        unsafe { ffi::set_default_options(&mut options); }
        Self {
            ffi: options
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

        let perm_r = ffi::intMalloc(m as c_int);
        assert!(!perm_r.is_null());

        let perm_c = ffi::intMalloc(n as c_int);
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
