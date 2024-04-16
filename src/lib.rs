use libc::{c_double, c_int};
use ndarray::{Array1, Array2};
use sprs::CsMat;
use std::mem;
use std::thread;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;
use superlu_sys as ffi;

use std::slice::from_raw_parts_mut;
use superlu_sys::{Dtype_t, Mtype_t, Stype_t};

mod tests;

#[derive(Debug)]
pub enum SolverError {
    Conflict,
    Unsolvable,
    Timeout,
}

pub struct Options {
    pub ffi: ffi::superlu_options_t,
}

impl Default for Options {
    fn default() -> Self {
        let mut options: ffi::superlu_options_t = unsafe { mem::zeroed() };
        unsafe {
            ffi::set_default_options(&mut options);
        }
        Self { ffi: options }
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

pub fn solve_super_lu(
    a: CsMat<f64>,
    b: &Vec<Array1<f64>>,
    timeout: Option<Duration>,
    options: &mut Options,
) -> Result<Vec<Array1<f64>>, SolverError> {
    let m = a.rows();
    let n = a.cols();
    if m != n {
        return Err(SolverError::Conflict);
    }
    if b.len() > 0 {
        if m != b[0].len() {
            return Err(SolverError::Conflict);
        }
        for rhs_col in b {
            if rhs_col.len() != b[0].len() {
                return Err(SolverError::Conflict);
            }
        }
    }
    if a.nnz() == 0 {
        return Err(SolverError::Unsolvable);
    }

    let a_mat = Arc::new(Mutex::new(SuperMatrix::from_csc_mat(a)));
    let b_mat = Arc::new(Mutex::new(SuperMatrix::from_ndarray(vec_of_array1_to_array2(b))));
    let options = Arc::new(Mutex::new(options.ffi));

    let a_mat_clone = Arc::clone(&a_mat);
    let b_mat_clone = Arc::clone(&b_mat);
    let options_clone = Arc::clone(&options);

    let (sender, receiver) = mpsc::channel();

    thread::spawn(move || unsafe {
        let perm_r = ffi::intMalloc(m as c_int);
        assert!(!perm_r.is_null());

        let perm_c = ffi::intMalloc(n as c_int);
        assert!(!perm_c.is_null());

        ffi::set_default_options(&mut *options_clone.lock().unwrap());

        let mut stat: ffi::SuperLUStat_t = mem::zeroed();
        ffi::StatInit(&mut stat);

        let mut l_mat: ffi::SuperMatrix = mem::zeroed();
        let mut u_mat: ffi::SuperMatrix = mem::zeroed();

        let mut info = 0;
        ffi::dgssv(
            &mut *options_clone.lock().unwrap(),
            a_mat_clone.lock().unwrap().raw_mut(),
            perm_c,
            perm_r,
            &mut l_mat,
            &mut u_mat,
            b_mat_clone.lock().unwrap().raw_mut(),
            &mut stat,
            &mut info,
        );

        ffi::SUPERLU_FREE(perm_r as *mut _);
        ffi::SUPERLU_FREE(perm_c as *mut _);
        ffi::Destroy_SuperNode_Matrix(&mut l_mat);
        ffi::Destroy_CompCol_Matrix(&mut u_mat);
        ffi::StatFree(&mut stat);

        if info != 0 {
            let _ = sender.send(Err(SolverError::Unsolvable));
        } else {
            let _ = sender.send(Ok(()));
        }
    });
    match timeout {
        None => {
            match receiver.recv() {
                Ok(res) => {
                    match res {
                        Ok(_) => {
                            let res_data = b_mat.lock().unwrap().raw().data_to_vec();
                            match res_data {
                                None => Err(SolverError::Unsolvable),
                                Some(data) => Ok(data
                                    .chunks(n)
                                    .map(|chunk| Array1::from_iter(chunk.iter().cloned()))
                                    .collect()),
                            }
                        },
                        Err(_) => Err(SolverError::Unsolvable)
                    }
                },
                Err(_) => {
                    panic!("Unknown internal SuperLU error");
                }
            }
        }
        Some(timeout_value) => {
            match receiver.recv_timeout(timeout_value) {
                Ok(res) => {
                    match res {
                        Ok(_) => {
                            let res_data = b_mat.lock().unwrap().raw().data_to_vec();
                            match res_data {
                                None => Err(SolverError::Unsolvable),
                                Some(data) => Ok(data
                                    .chunks(n)
                                    .map(|chunk| Array1::from_iter(chunk.iter().cloned()))
                                    .collect()),
                            }
                        }
                        Err(_) => {Err(SolverError::Unsolvable)}
                    }
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    return Err(SolverError::Timeout);
                },
                Err(_) => {
                    panic!("Unknown internal SuperLU error");
                }
            }
        }
    }

}

pub struct SuperMatrix {
    raw: ffi::SuperMatrix,
    rust_managed: bool,
}

pub trait FromSuperMatrix: Sized {
    fn from_super_matrix(_: &SuperMatrix) -> Option<Self>;
}

unsafe impl Send for SuperMatrix {}

impl SuperMatrix {
    pub unsafe fn from_raw(raw: ffi::SuperMatrix) -> SuperMatrix {
        SuperMatrix {
            raw,
            rust_managed: false,
        }
    }

    pub fn into_raw(self) -> ffi::SuperMatrix {
        let raw = self.raw;
        if self.rust_managed {
            mem::forget(self);
        }
        raw
    }

    pub fn from_csc_mat(mat: CsMat<f64>) -> Self {
        assert_eq!(mat.storage(), sprs::CompressedStorage::CSC);

        let m = mat.rows() as c_int;
        let n = mat.cols() as c_int;
        let nnz = mat.nnz() as c_int;

        let mut raw: ffi::SuperMatrix = unsafe { mem::zeroed() };

        let nzval: Vec<c_double> = mat.data().iter().map(|&x| x as c_double).collect();
        let rowind: Vec<c_int> = mat.indices().iter().map(|&x| x as c_int).collect();
        let mut colptr = Vec::new();
        let colptr_raw = mat.indptr();
        for ptr in colptr_raw.as_slice().unwrap() {
            colptr.push(ptr.clone() as c_int)
        }

        let nzval_boxed = nzval.into_boxed_slice();
        let rowind_boxed = rowind.into_boxed_slice();
        let colptr_boxed = colptr.into_boxed_slice();

        let nzval_ptr = Box::leak(nzval_boxed).as_mut_ptr();
        let rowind_ptr = Box::leak(rowind_boxed).as_mut_ptr();
        let colptr_ptr = Box::leak(colptr_boxed).as_mut_ptr();

        unsafe {
            ffi::dCreate_CompCol_Matrix(
                &mut raw,
                m,
                n,
                nnz,
                nzval_ptr as *mut c_double,
                rowind_ptr as *mut c_int,
                colptr_ptr as *mut c_int,
                Stype_t::SLU_NC,
                Dtype_t::SLU_D,
                Mtype_t::SLU_GE,
            );
        }
        unsafe { Self::from_raw(raw) }
    }

    pub fn from_ndarray(array: Array2<f64>) -> Self {
        let nrows = array.nrows() as c_int;
        let ncols = array.ncols() as c_int;

        let col_major_data = unsafe { ffi::doubleMalloc(ncols * nrows) };
        let mut index: usize = 0;
        let col_major_data_ptr =
            unsafe { from_raw_parts_mut(col_major_data, (ncols * nrows) as usize) };
        for col in 0..ncols as usize {
            for row in 0..nrows as usize {
                col_major_data_ptr[index] = array[[row, col]];
                index += 1;
            }
        }

        let mut raw: ffi::SuperMatrix = unsafe { std::mem::zeroed() };

        unsafe {
            ffi::dCreate_Dense_Matrix(
                &mut raw as *mut ffi::SuperMatrix,
                nrows,
                ncols,
                col_major_data,
                nrows,
                Stype_t::SLU_DN,
                Dtype_t::SLU_D,
                Mtype_t::SLU_GE,
            );

            SuperMatrix {
                raw,
                rust_managed: true,
            }
        }
    }

    pub fn into_ndarray(self) -> Option<Array2<f64>> {
        match self.raw.data_to_vec() {
            None => None,
            Some(data) => match Array2::from_shape_vec((self.nrows(), self.ncols()), data) {
                Ok(arr) => Some(arr.t().to_owned()),
                Err(_) => None,
            },
        }
    }

    pub fn nrows(&self) -> usize {
        self.raw.nrow as usize
    }

    pub fn ncols(&self) -> usize {
        self.raw.ncol as usize
    }

    pub fn raw(&self) -> &ffi::SuperMatrix {
        &self.raw
    }

    pub fn raw_mut(&mut self) -> *mut ffi::SuperMatrix {
        &mut self.raw
    }
}

impl Drop for SuperMatrix {
    fn drop(&mut self) {
        unsafe {
            let store = &*(self.raw().Store as *const ffi::NCformat);
            if store.nnz == 0 {
                return;
            }
        }
        if self.rust_managed {
            match self.raw.Stype {
                Stype_t::SLU_NC => unsafe {
                    ffi::Destroy_CompCol_Matrix(&mut self.raw);
                },
                Stype_t::SLU_NCP => unsafe {
                    ffi::Destroy_CompCol_Permuted(&mut self.raw);
                },
                Stype_t::SLU_NR => unsafe {
                    ffi::Destroy_CompRow_Matrix(&mut self.raw);
                },
                Stype_t::SLU_SC | ffi::Stype_t::SLU_SCP | ffi::Stype_t::SLU_SR => unsafe {
                    ffi::Destroy_SuperNode_Matrix(&mut self.raw);
                },
                Stype_t::SLU_DN => unsafe {
                    ffi::Destroy_Dense_Matrix(&mut self.raw);
                },
                _ => {}
            }
        }
    }
}
