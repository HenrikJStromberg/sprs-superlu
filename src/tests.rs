#[cfg(test)]
mod super_matrix_tests {
    use crate::SuperMatrix;
    use ndarray::{arr2, Array2};
    use sprs::{CsMat, TriMat};
    use std::mem;
    use superlu_sys::{Dtype_t, Mtype_t, Stype_t};

    extern crate superlu_sys as ffi;

    fn array2s_close(a: &Array2<f64>, b: &Array2<f64>, criterion: f64) -> bool {
        if a.nrows() != b.nrows() {
            return false;
        };
        if a.ncols() != b.ncols() {
            return false;
        };
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                if ((a[[i, j]] - b[[i, j]]) / a[[i, j]]).abs() > criterion {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_from_csc_mat_basic() {
        use ffi::colperm_t::*;

        let values = vec![
            19.0, 12.0, 12.0, 21.0, 12.0, 12.0, 21.0, 16.0, 21.0, 5.0, 21.0, 18.0,
        ];
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

            let mut options: ffi::superlu_options_t = mem::zeroed();
            ffi::set_default_options(&mut options);
            options.ColPerm = NATURAL;

            let mut stat: ffi::SuperLUStat_t = mem::zeroed();
            ffi::StatInit(&mut stat);

            let mut l_mat: ffi::SuperMatrix = mem::zeroed();
            let mut u_mat: ffi::SuperMatrix = mem::zeroed();

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
            if info != 0 {
                panic!("solver error")
            }
            let res = b_mat.raw().data_to_vec();
            ffi::SUPERLU_FREE(perm_r as *mut _);
            ffi::SUPERLU_FREE(perm_c as *mut _);
            ffi::Destroy_SuperNode_Matrix(&mut l_mat);
            ffi::Destroy_CompCol_Matrix(&mut u_mat);
            ffi::StatFree(&mut stat);
            res.unwrap()
        };

        let sol = arr2(&[[-0.03125, 0.065476, 0.013393, 0.0625, 0.032738]])
            .t()
            .to_owned();
        assert!(array2s_close(
            &Array2::from_shape_vec((5, 1), res).unwrap(),
            &sol,
            1e-4
        ));
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
        use superlu_sys::{Dtype_t, Mtype_t, Stype_t};

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
            _ => {
                panic!("Stype!=SLU_DN")
            }
        }
        match super_matrix.raw().Dtype {
            Dtype_t::SLU_D => {}
            _ => {
                panic!("Dtype!=SLU_D")
            }
        }
        match super_matrix.raw().Mtype {
            Mtype_t::SLU_GE => {}
            _ => {
                panic!("Mtype!=SLU_GE")
            }
        }
        assert!(!super_matrix.raw().Store.is_null());
    }
}

#[cfg(test)]
mod solver_tests {
    use crate::SolverError;
    use crate::{solve_super_lu, Options};
    use ndarray::{arr1, Array1};
    use sprs::{CsMat, TriMat};
    use std::time::Duration;

    extern crate superlu_sys as ffi;

    fn array1s_close(a: &Array1<f64>, b: &Array1<f64>, criterion: f64) -> bool {
        if a.len() != b.len() {
            return false;
        };
        for i in 0..a.len() {
            if ((a[i] - b[i]) / a[i]).abs() > criterion {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_solver() {
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
        let res = solve_super_lu(a_mat, &b_mat, Some(Duration::from_secs(5)), &mut options);

        let expected_vec = arr1(&[
            -0.03125000000000001,
            0.06547619047619048,
            0.013392857142857147,
            0.0625,
            0.03273809523809524,
        ]);
        let expected = vec![expected_vec.clone(), expected_vec.clone() * 2.];

        match res {
            Ok(sol) => {
                assert_eq!(b_mat.len(), sol.len());
                for i in 0..b_mat.len() {
                    assert!(array1s_close(&sol[i], &expected[i], 1e-4));
                }
            }
            Err(_) => {
                panic!("internal solver error")
            }
        }
    }

    #[test]
    fn test_no_timeout() {
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
        let res = solve_super_lu(a_mat, &b_mat, None, &mut options);

        let expected_vec = arr1(&[
            -0.03125000000000001,
            0.06547619047619048,
            0.013392857142857147,
            0.0625,
            0.03273809523809524,
        ]);
        let expected = vec![expected_vec.clone(), expected_vec.clone() * 2.];

        match res {
            Ok(sol) => {
                assert_eq!(b_mat.len(), sol.len());
                for i in 0..b_mat.len() {
                    assert!(array1s_close(&sol[i], &expected[i], 1e-4));
                }
            }
            Err(_) => {
                panic!("internal solver error")
            }
        }
    }

    #[test]
    fn test_timeout() {
        let size = 10000;

        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptrs = Vec::with_capacity(size + 1);

        col_ptrs.push(0);

        for i in 0..size {
            let mut current_column_values = Vec::new();
            let mut current_column_indices = Vec::new();

            current_column_values.push(10.0);
            current_column_indices.push(i);

            if i > 0 {
                current_column_values.push(1.0);
                current_column_indices.push(i - 1);
            }

            if i < size - 1 {
                current_column_values.push(1.0);
                current_column_indices.push(i + 1);
            }

            let mut combined: Vec<_> = current_column_indices
                .into_iter()
                .zip(current_column_values.into_iter())
                .collect();
            combined.sort_by_key(|&(index, _)| index);

            let (sorted_indices, sorted_values): (Vec<_>, Vec<_>) = combined.into_iter().unzip();

            row_indices.extend(sorted_indices);
            values.extend(sorted_values);
            col_ptrs.push(values.len());
        }

        let a_mat = CsMat::new_csc((size, size), col_ptrs, row_indices, values);

        let rhs_1 = arr1(&vec![1.0; size]);
        let rhs_2 = arr1(&vec![2.0; size]);
        let b_mat = vec![rhs_1, rhs_2];

        let mut options = Options::default();

        let res = solve_super_lu(a_mat, &b_mat, Some(Duration::from_nanos(1)), &mut options);

        match res {
            Ok(_) => {
                panic!("Timeout not caught");
            }
            Err(e) => match e {
                SolverError::Timeout => {}
                _ => panic!("Unexpected error: {:?}", e),
            },
        }
    }

    #[test]
    fn test_solver_singular_matrix() {
        let mut tri_mat = TriMat::new((5, 5));
        tri_mat.add_triplet(0, 0, 1.0);
        tri_mat.add_triplet(1, 0, 1.0);
        tri_mat.add_triplet(2, 2, 1.0);
        tri_mat.add_triplet(3, 3, 1.0);
        tri_mat.add_triplet(4, 0, 1.0);
        tri_mat.add_triplet(4, 1, 1.0);
        tri_mat.add_triplet(4, 2, 1.0);
        tri_mat.add_triplet(4, 3, 1.0);

        let a_mat: CsMat<f64> = tri_mat.to_csc();
        let b_mat = vec![arr1(&[1., 1., 1., 1., 1.])];
        let mut options = Options::default();
        let res = solve_super_lu(a_mat, &b_mat, Some(Duration::from_secs(5)), &mut options);
        match res {
            Ok(_) => {
                panic!("Singular matrix to caught");
            }
            Err(e) => match e {
                SolverError::Unsolvable => {}
                _ => {
                    panic!("Singular matrix to caught");
                }
            },
        }
    }

    #[test]
    fn test_solver_all_zero_matrix() {
        let tri_mat = TriMat::new((5, 5));
        let a_mat: CsMat<f64> = tri_mat.to_csc();
        let b_mat = vec![arr1(&[1., 1., 1., 1., 1.])];
        let mut options = Options::default();
        let res = solve_super_lu(a_mat, &b_mat, Some(Duration::from_secs(5)), &mut options);
        match res {
            Ok(_) => {
                panic!("Singular matrix to caught");
            }
            Err(e) => match e {
                SolverError::Unsolvable => {}
                _ => {
                    panic!("Singular matrix to caught");
                }
            },
        }
    }

    #[test]
    fn test_solver_matrix_mismatch() {
        let a_mat: CsMat<f64> = TriMat::new((5, 5)).to_csc();
        let b_mat = vec![arr1(&[1., 1., 1., 1.])];
        let mut options = Options::default();
        let res = solve_super_lu(a_mat, &b_mat, Some(Duration::from_secs(5)), &mut options);
        match res {
            Ok(_) => {
                panic!("Dimension error to caught");
            }
            Err(e) => match e {
                SolverError::Conflict => {}
                _ => {
                    panic!("Dimension error to caught");
                }
            },
        }
    }

    #[test]
    fn test_solver_rhs_mismatch() {
        let a_mat: CsMat<f64> = TriMat::new((5, 5)).to_csc();
        let b_mat = vec![arr1(&[1., 1., 1., 1., 1.]), arr1(&[1., 1., 1., 1.])];
        let mut options = Options::default();
        let res = solve_super_lu(a_mat, &b_mat, Some(Duration::from_secs(5)), &mut options);
        match res {
            Ok(_) => {
                panic!("Dimension error to caught");
            }
            Err(e) => match e {
                SolverError::Conflict => {}
                _ => {
                    panic!("Dimension error to caught");
                }
            },
        }
    }
}

#[cfg(test)]
mod multiplication_tests {
    use crate::SuperMatrix;
    use ndarray::array;
    use sprs::CsMat;

    extern crate superlu_sys as ffi;
    #[test]
    fn test_dot_vct_basic() {
        let data = vec![1.0, 2.0, 3.0];
        let indices = vec![0, 1, 2];
        let indptr = vec![0, 1, 2, 3];

        let csc_matrix = CsMat::new_csc((3, 3), indptr, indices, data);
        let mut a = SuperMatrix::from_csc_mat(csc_matrix);

        let vector_b = array![1.0, 1.0, 1.0];

        let result = a.dot_vct(vector_b.view(), false).unwrap();

        assert_eq!(result, array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dot_vct_transposed() {
        let data = vec![1.0, 2.0];
        let indices = vec![0, 1];
        let indptr = vec![0, 1, 2, 2];

        let csc_matrix = CsMat::new_csc((2, 3), indptr, indices, data);
        let mut a = SuperMatrix::from_csc_mat(csc_matrix);

        let vector_b = array![1.0, 1.0];

        let result = a.dot_vct(vector_b.view(), true).unwrap();

        assert_eq!(result, array![1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_dot_vct_dimension_mismatch() {
        let data = vec![1.0, 2.0];
        let indices = vec![0, 1];
        let indptr = vec![0, 1, 2];

        let csc_matrix = CsMat::new_csc((2, 2), indptr, indices, data);
        let mut a = SuperMatrix::from_csc_mat(csc_matrix);

        let vector_b = array![1.0];

        let err = a.dot_vct(vector_b.view(), false).unwrap_err();
        assert_eq!(
            err,
            "Input vector length must be equal to the number of columns when not transposed."
                .to_string()
        )
    }

    #[test]
    fn test_dot_vct_zero_vector() {
        let data = vec![4.0, 5.0, 6.0];
        let indices = vec![0, 1, 2];
        let indptr = vec![0, 1, 2, 3];

        let csc_matrix = CsMat::new_csc((3, 3), indptr, indices, data);
        let mut a = SuperMatrix::from_csc_mat(csc_matrix);

        let vector_b = array![0.0, 0.0, 0.0];

        let result = a.dot_vct(vector_b.view(), false).unwrap();

        assert_eq!(result, array![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dot_vct_dense_matrix() {
        let data = vec![1.0, 2.0, 4.0, 3.0];
        let indices = vec![0, 1, 0, 2];
        let indptr = vec![0, 1, 2, 4];

        let csc_matrix = CsMat::new_csc((3, 3), indptr, indices, data);

        let mut a = SuperMatrix::from_csc_mat(csc_matrix);

        let vector_b = array![1.0, 1.0, 1.0];

        let result = a.dot_vct(vector_b.view(), false).unwrap();

        assert_eq!(result, array![5.0, 2.0, 3.0]);
    }
}
