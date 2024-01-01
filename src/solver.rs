use sprs::CsMat;
use ndarray::Array2;
use superlu_sys::{superlu_options_t, yes_no_t};
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

/*
pub fn solve (a: CsMat<f64>, b: Array2<f64>, options: Options) -> Array2<f64> {

}
 */