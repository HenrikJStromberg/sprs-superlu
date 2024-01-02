use std::mem::MaybeUninit;
use std::slice::from_raw_parts_mut;
use sprs::CsMat;
use ndarray::{Array1, Array2, Axis};
use superlu_sys::{superlu_options_t, yes_no_t};
use superlu_sys;
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

pub fn solve (a: CsMat<f64>, b: &Vec<Array1<f64>>, options: &Options) -> Result<Vec<Array1<f64>>, SolverError> {
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
    let a_super_matrix = SuperMatrix::from_csc_mat(a);
    //println!("{:?}", a_super_matrix.raw().Store.);
    //let rhs = SuperMatrix::from_ndarray(Array2::stack(Axis(0), b.iter().map(|col| col.view())).unwrap());
    //let mut rhs_raw = rhs.into_raw();

    let res_data = unsafe {
        let (m, n, nnz) = (5, 5, 12);
        /*
        let a = superlu_sys::doubleMalloc(nnz);
        assert!(!a.is_null());
        {
            let (s, u, p, e, r, l) = (19.0, 21.0, 16.0, 5.0, 18.0, 12.0);
            let a = from_raw_parts_mut(a, nnz as usize);
            a[0] = s;
            a[1] = l;
            a[2] = l;
            a[3] = u;
            a[4] = l;
            a[5] = l;
            a[6] = u;
            a[7] = p;
            a[8] = u;
            a[9] = e;
            a[10] = u;
            a[11] = r;
        }

        let asub = superlu_sys::intMalloc(nnz);
        assert!(!asub.is_null());
        {
            let asub = from_raw_parts_mut(asub, nnz as usize);
            asub[0] = 0;
            asub[1] = 1;
            asub[2] = 4;
            asub[3] = 1;
            asub[4] = 2;
            asub[5] = 4;
            asub[6] = 0;
            asub[7] = 2;
            asub[8] = 0;
            asub[9] = 3;
            asub[10] = 3;
            asub[11] = 4;
        }

        let xa = superlu_sys::intMalloc(n + 1);
        assert!(!xa.is_null());
        {
            let xa = from_raw_parts_mut(xa, (n + 1) as usize);
            xa[0] = 0;
            xa[1] = 3;
            xa[2] = 6;
            xa[3] = 8;
            xa[4] = 10;
            xa[5] = 12;
        }

        let mut A: superlu_sys::SuperMatrix = MaybeUninit::zeroed().assume_init();

        superlu_sys::dCreate_CompCol_Matrix(&mut A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);


         */
        let mut A = a_super_matrix.into_raw();

        let nrhs = 1;
        let rhs = superlu_sys::doubleMalloc(m * nrhs);
        assert!(!rhs.is_null());
        {
            let rhs = from_raw_parts_mut(rhs, (m * nrhs) as usize);
            for i in 0..((m * nrhs) as usize) {
                rhs[i] = 1.0;
            }
        }

        let mut B: superlu_sys::SuperMatrix = MaybeUninit::zeroed().assume_init();
        superlu_sys::dCreate_Dense_Matrix(&mut B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);

        let perm_r = superlu_sys::intMalloc(m);
        assert!(!perm_r.is_null());

        let perm_c = superlu_sys::intMalloc(n);
        assert!(!perm_c.is_null());

        let mut options: superlu_options_t = MaybeUninit::zeroed().assume_init();
        superlu_sys::set_default_options(&mut options);
        options.ColPerm = superlu_sys::colperm_t::NATURAL;

        let mut stat: superlu_sys::SuperLUStat_t = MaybeUninit::zeroed().assume_init();
        superlu_sys::StatInit(&mut stat);

        let mut L: superlu_sys::SuperMatrix = MaybeUninit::zeroed().assume_init();
        let mut U: superlu_sys::SuperMatrix = MaybeUninit::zeroed().assume_init();

        let mut info = 0;
        superlu_sys::dgssv(
            &mut options,
            &mut A,
            perm_c,
            perm_r,
            &mut L,
            &mut U,
            &mut B,
            &mut stat,
            &mut info,
        );
        if info != 0 {return Err(Diverged)}
        let res_data = B.data_as_vec().expect("internal solver error");
        superlu_sys::SUPERLU_FREE(rhs as *mut _);
        superlu_sys::SUPERLU_FREE(perm_r as *mut _);
        superlu_sys::SUPERLU_FREE(perm_c as *mut _);
        superlu_sys::Destroy_CompCol_Matrix(&mut A);
        superlu_sys::Destroy_SuperMatrix_Store(&mut B);
        superlu_sys::Destroy_SuperNode_Matrix(&mut L);
        superlu_sys::Destroy_CompCol_Matrix(&mut U);
        superlu_sys::StatFree(&mut stat);
        res_data
    };

    Ok(res_data
        .chunks(n)
        .map(|chunk| Array1::from_iter(chunk.iter().cloned()))
        .collect())
}
