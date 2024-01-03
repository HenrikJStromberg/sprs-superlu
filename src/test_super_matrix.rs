#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;
    use std::slice::from_raw_parts_mut;
    use crate::super_matrix::SuperMatrix;
    use sprs::{CsMat, TriMat};
    use ndarray::{arr1, arr2, Array1, Array2};
    use superlu_sys::{Dtype_t, Mtype_t, Stype_t};
    use superlu_sys::colperm_t::NATURAL;
    use crate::solver::{Options, solve};
    use crate::solver::SolverError::Diverged;

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
        use ffi::Dtype_t::*;
        use ffi::Mtype_t::*;
        use ffi::Stype_t::*;
        use ffi::colperm_t::*;
        use ffi::Dtype_t::SLU_D;
        use ffi::Mtype_t::SLU_GE;
        use ffi::Stype_t::{SLU_DN, SLU_NC};

        let values = vec![19.0, 12.0, 12.0, 21.0, 12.0, 12.0, 21.0, 16.0, 21.0, 5.0, 21.0, 18.0];
        let row_indices = vec![0, 1, 4, 1, 2, 4, 0, 2, 0, 3, 3, 4];
        let col_ptrs = vec![0, 3, 6, 8, 10, 12];
        let a_csc = CsMat::new_csc((5, 5), col_ptrs, row_indices, values);
        let mut a_mat = SuperMatrix::from_csc_mat(a_csc.clone());

        let mut b_mat = SuperMatrix::from_ndarray(arr2(&[[1., 1., 1., 1., 1.]]).t().to_owned());
        let res = unsafe {

            let (m, n, nnz) = (5, 5, 12);

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

    //ToDo: test for conflicting dimensions
    //ToDo: test for singular matrix

    #[test]
    fn test_solver() {
        let values = vec![19.0, 12.0, 12.0, 21.0, 12.0, 12.0, 21.0, 16.0, 21.0, 5.0, 21.0, 18.0];
        let row_indices = vec![0, 1, 4, 1, 2, 4, 0, 2, 0, 3, 3, 4];
        let col_ptrs = vec![0, 3, 6, 8, 10, 12];
        let a_mat = CsMat::new_csc((5, 5), col_ptrs, row_indices, values);
        let b_mat = vec![arr1(&[1., 1., 1., 1., 1.])];
        let mut options = Options::default();
        options.ffi.ColPerm = NATURAL;
        let res = solve(a_mat, &b_mat, &mut options);

        let expected = vec![arr1(&[-0.03125, 0.065476, 0.013393, 0.0625, 0.032738])];

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

        unsafe {
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
        }
        let back_conversion = super_matrix.into_ndarray().unwrap();
        assert!(array2s_close(&array, &back_conversion, 0.01));
    }


    #[test]
    fn test_from_ndarray_empty() {
        let array = Array2::<f64>::zeros((2, 2));

        let super_matrix = SuperMatrix::from_ndarray(array);

        unsafe {
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

    #[test]
    fn test_from_raw() {
        let mat = unsafe {
            let (m, n, nnz) = (5, 5, 12);

            let a = ffi::doubleMalloc(nnz);
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

            let asub = ffi::intMalloc(nnz);
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

            let xa = ffi::intMalloc(n + 1);
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

            let mut mat: ffi::SuperMatrix = MaybeUninit::zeroed().assume_init();

            ffi::dCreate_CompCol_Matrix(&mut mat, m, n, nnz, a, asub, xa, Stype_t::SLU_NC, Dtype_t::SLU_D, Mtype_t::SLU_GE);
            mat
        };
        let super_mat = unsafe {SuperMatrix::from_raw(mat)};
    }
}
