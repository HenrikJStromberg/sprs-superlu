#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;
    use crate::SuperMatrix;
    use sprs::{CsMat, TriMat};
    use ndarray::{arr1, arr2, Array1, Array2};
    use superlu_sys::{Dtype_t, Mtype_t, Stype_t};
    use crate::{Options, solve_super_lu};
    use crate::SolverError;

    extern crate superlu_sys as ffi;

    fn array2s_close(a: &Array2<f64>, b: &Array2<f64>, criterion: f64) -> bool {
        if a.nrows() != b.nrows() {return false};
        if a.ncols() != b.ncols() {return false};
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                if ((a[[i, j]] - b[[i, j]]) / a[[i, j]]).abs() > criterion {
                    return false
                }
            }
        }
        true
    }

    fn array1s_close(a: &Array1<f64>, b: &Array1<f64>, criterion: f64) -> bool {
        if a.len() != b.len() {return false};
        for i in 0..a.len() {
            if ((a[i] - b[i]) / a[i]).abs() > criterion {
                return false
            }
        }
        true
    }

    #[test]
    fn test_from_csc_mat_basic() {
        use ffi::colperm_t::*;

        let values = vec![19.0, 12.0, 12.0, 21.0, 12.0, 12.0, 21.0, 16.0, 21.0, 5.0, 21.0, 18.0];
        let row_indices = vec![0, 1, 4, 1, 2, 4, 0, 2, 0, 3, 3, 4];
        let col_ptrs = vec![0, 3, 6, 8, 10, 12];
        let a_csc = CsMat::new_csc((5, 5), col_ptrs, row_indices, values);
        let mut a_mat = SuperMatrix::from_csc_mat(a_csc.clone());

        let mut b_mat = SuperMatrix::from_ndarray(arr2(&[[1., 1., 1., 1., 1.]]).t().to_owned());
        let res = unsafe {

            let (m, n) = (5, 5);

            let perm_r = ffi::intMalloc(m);
            assert!(!perm_r.is_null());

            let perm_c = ffi::intMalloc(n);
            assert!(!perm_c.is_null());

            let mut options: ffi::superlu_options_t = MaybeUninit::zeroed().assume_init();
            ffi::set_default_options(&mut options);
            options.ColPerm = NATURAL;

            let mut stat: ffi::SuperLUStat_t = MaybeUninit::zeroed().assume_init();
            ffi::StatInit(&mut stat);

            let mut l_mat: ffi::SuperMatrix = MaybeUninit::zeroed().assume_init();
            let mut u_mat: ffi::SuperMatrix = MaybeUninit::zeroed().assume_init();

            let mut info = 0;
            ffi::dgssv(
                &mut options,
                a_mat.raw_mut(),
                perm_c,
                perm_r,
                &mut l_mat,
                &mut u_mat,
                b_mat.raw_mut(),
                &mut stat,
                &mut info,
            );
            if info != 0 {panic!("solver error")}
            let res = b_mat.raw().data_as_vec();
            ffi::SUPERLU_FREE(perm_r as *mut _);
            ffi::SUPERLU_FREE(perm_c as *mut _);
            ffi::Destroy_SuperNode_Matrix(&mut l_mat);
            ffi::Destroy_CompCol_Matrix(&mut u_mat);
            ffi::StatFree(&mut stat);
            res.unwrap()
        };

        let sol = arr2(&[[-0.03125, 0.065476, 0.013393, 0.0625, 0.032738]]).t().to_owned();
        assert!(array2s_close(&Array2::from_shape_vec((5, 1), res).unwrap(), &sol, 1e-4));
    }

    #[test]
    fn test_solver() {
        let values = vec![19.0, 12.0, 12.0, 21.0, 12.0, 12.0, 21.0, 16.0, 21.0, 5.0, 21.0, 18.0];
        let row_indices = vec![0, 1, 4, 1, 2, 4, 0, 2, 0, 3, 3, 4];
        let col_ptrs = vec![0, 3, 6, 8, 10, 12];
        let a_mat = CsMat::new_csc((5, 5), col_ptrs, row_indices, values);
        let b_mat = vec![arr1(&[1., 1., 1., 1., 1.]),
                                            arr1(&[2., 2., 2., 2., 2.])];
        let mut options = Options::default();
        let res = solve_super_lu(a_mat, &b_mat, &mut options);

        let expected_vec = arr1(&[-0.03125000000000001, 0.06547619047619048,
            0.013392857142857147, 0.0625, 0.03273809523809524]);
        let expected = vec![expected_vec.clone(), expected_vec.clone() * 2.];

        match res {
            Ok(sol) => {
                assert_eq!(b_mat.len(), sol.len());
                for i in 0..b_mat.len() {
                    assert!(array1s_close(&sol[i], &expected[i], 1e-4));
                }
            }
            Err(_) => {panic!("internal solver error")}
        }
    }


    #[test]
    fn test_solver_singular_matrix() {
        //ToDo: fix random failure of test inside dgssv
        let a_mat: CsMat<f64> = TriMat::new((5, 5)).to_csc();
        let b_mat = vec![arr1(&[1., 1., 1., 1., 1.])];
        let mut options = Options::default();
        let res = solve_super_lu(a_mat, &b_mat, &mut options);
        match res {
            Ok(_) => {
                panic!("Singular matrix to caught");
            }
            Err(e) => {
                match e {
                    SolverError::Diverged => {}
                    _ => {panic!("Singular matrix to caught");}
                }
            }
        }
    }

    //ToDo: test that options are actually used

    #[test]
    fn test_solver_matrix_mismatch() {
        let a_mat: CsMat<f64> = TriMat::new((5, 5)).to_csc();
        let b_mat = vec![arr1(&[1., 1., 1., 1.])];
        let mut options = Options::default();
        let res = solve_super_lu(a_mat, &b_mat, &mut options);
        match res {
            Ok(_) => {
                panic!("Dimension error to caught");
            }
            Err(e) => {
                match e {
                    SolverError::Conflict => {}
                    _ => {panic!("Dimension error to caught");}
                }
            }
        }
    }

    #[test]
    fn test_solver_rhs_mismatch() {
        let a_mat: CsMat<f64> = TriMat::new((5, 5)).to_csc();
        let b_mat = vec![arr1(&[1., 1., 1., 1., 1.]), arr1(&[1., 1., 1., 1.])];
        let mut options = Options::default();
        let res = solve_super_lu(a_mat, &b_mat, &mut options);
        match res {
            Ok(_) => {
                panic!("Dimension error to caught");
            }
            Err(e) => {
                match e {
                    SolverError::Conflict => {}
                    _ => {panic!("Dimension error to caught");}
                }
            }
        }
    }

    #[test]
    fn test_from_csc_mat_empty() {
        let tri_mat = TriMat::new((3, 3));
        let csc_mat: CsMat<f64> = tri_mat.to_csc();

        let super_matrix = SuperMatrix::from_csc_mat(csc_mat);

        unsafe {
            assert_eq!(super_matrix.nrows(), 3);
            assert_eq!(super_matrix.ncols(), 3);
            let store = &*(super_matrix.raw().Store as *const ffi::NCformat);
            assert_eq!(store.nnz, 0);
        }
    }

    #[test]
    fn test_from_ndarray_basic() {
        use superlu_sys::{Stype_t, Dtype_t, Mtype_t};

        let array = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let super_matrix = SuperMatrix::from_ndarray(array.clone());

        assert_eq!(super_matrix.nrows(), 2);
        assert_eq!(super_matrix.ncols(), 2);
        match super_matrix.raw().Stype {
            Stype_t::SLU_DN => {}
            _ => panic!("Stype != SLU_DN"),
        }
        match super_matrix.raw().Dtype {
            Dtype_t::SLU_D => {}
            _ => panic!("Dtype != SLU_D"),
        }
        match super_matrix.raw().Mtype {
            Mtype_t::SLU_GE => {}
            _ => panic!("Mtype != SLU_GE"),
        }
        let back_conversion = super_matrix.into_ndarray().unwrap();
        assert!(array2s_close(&array, &back_conversion, 0.01));
    }


    #[test]
    fn test_from_ndarray_empty() {
        let array = Array2::<f64>::zeros((2, 2));

        let super_matrix = SuperMatrix::from_ndarray(array);

        assert_eq!(super_matrix.nrows(), 2);
        assert_eq!(super_matrix.ncols(), 2);
        match super_matrix.raw().Stype {
            Stype_t::SLU_DN => {}
            _ => {panic!("Stype!=SLU_DN")}
        }
        match super_matrix.raw().Dtype {
            Dtype_t::SLU_D => {}
            _ => {panic!("Dtype!=SLU_D")}
        }
        match super_matrix.raw().Mtype {
            Mtype_t::SLU_GE => {}
            _ => {panic!("Mtype!=SLU_GE")}
        }
        assert!(!super_matrix.raw().Store.is_null());
    }
}
