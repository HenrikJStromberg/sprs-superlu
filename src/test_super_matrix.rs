#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;
    use std::slice::from_raw_parts_mut;
    use crate::super_matrix::SuperMatrix;
    use sprs::{CsMat, TriMat};
    use ndarray::{arr1, Array2};
    use superlu_sys::{Dtype_t, Mtype_t, Stype_t};

    extern crate superlu_sys as ffi;

    fn arrays_close(a: Array2<f64>, b: Array2<f64>, criterion: f64) -> bool {
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

    #[test]
    fn test_from_csc_mat_basic() {
        let values = vec![19.0, 12.0, 12.0, 21.0, 12.0, 12.0, 21.0, 16.0, 21.0, 5.0, 21.0, 18.0];
        let row_indices = vec![0, 1, 4, 1, 2, 4, 0, 2, 0, 3, 3, 4];

        let col_ptrs = vec![0, 3, 6, 8, 10, 12];
        let A = SuperMatrix::from_csc_mat(CsMat::new_csc((5, 5), col_ptrs, row_indices, values));
        let B = vec![arr1(&[1., 1., 1., 1., 1.])];
        assert_eq!(A.nrows(), 5);
        assert_eq!(A.ncols(), 5);
        //ToDo: assert correctness
    }

    #[test]
    fn test_raw_solver() {
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
        let A_csc = CsMat::new_csc((5, 5), col_ptrs, row_indices, values);
        let mut A = SuperMatrix::from_csc_mat(A_csc.clone());

        unsafe {

            let (m, n, nnz) = (5, 5, 12);

            let nrhs = 1;
            let rhs = ffi::doubleMalloc(m * nrhs);
            assert!(!rhs.is_null());
            {
                let rhs = from_raw_parts_mut(rhs, (m * nrhs) as usize);
                for i in 0..((m * nrhs) as usize) {
                    rhs[i] = 1.0;
                }
            }

            let mut B: ffi::SuperMatrix = MaybeUninit::zeroed().assume_init();
            ffi::dCreate_Dense_Matrix(&mut B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);

            let perm_r = ffi::intMalloc(m);
            assert!(!perm_r.is_null());

            let perm_c = ffi::intMalloc(n);
            assert!(!perm_c.is_null());

            let mut options: ffi::superlu_options_t = MaybeUninit::zeroed().assume_init();
            ffi::set_default_options(&mut options);
            options.ColPerm = NATURAL;

            let mut stat: ffi::SuperLUStat_t = MaybeUninit::zeroed().assume_init();
            ffi::StatInit(&mut stat);

            let mut L: ffi::SuperMatrix = MaybeUninit::zeroed().assume_init();
            let mut U: ffi::SuperMatrix = MaybeUninit::zeroed().assume_init();

            let mut info = 0;
            ffi::dgssv(
                &mut options,
                A.raw_mut(),
                perm_c,
                perm_r,
                &mut L,
                &mut U,
                &mut B,
                &mut stat,
                &mut info,
            );

            let v = B.data_as_vec();
            println!("v: {:?}", v);
            ffi::SUPERLU_FREE(rhs as *mut _);
            ffi::SUPERLU_FREE(perm_r as *mut _);
            ffi::SUPERLU_FREE(perm_c as *mut _);
            ffi::Destroy_SuperMatrix_Store(&mut B);
            ffi::Destroy_SuperNode_Matrix(&mut L);
            ffi::Destroy_CompCol_Matrix(&mut U);
            ffi::StatFree(&mut stat);
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
        assert!(arrays_close(array, back_conversion, 0.01));
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
