//! Wrappers for [SuperLU].
//!
//! [superlu]: http://crd-legacy.lbl.gov/~xiaoye/SuperLU

extern crate libc;
extern crate matrix;
extern crate superlu_sys as ffi;

use std::mem::MaybeUninit;
use ndarray::{Array2, ArrayBase, ArrayView2, Dim, Ix2, OwnedRepr, ShapeError};
use matrix::format::Compressed;
use std::mem;
use std::os::raw::{c_int, c_double};
use std::slice::from_raw_parts_mut;
use sprs::CsMat;
use superlu_sys::{Dtype_t, Mtype_t, Stype_t};

/// A super matrix.
pub struct SuperMatrix {
    raw: ffi::SuperMatrix,
    rust_managed: bool,
}

/// A type capable of instantiating itself from a super matrix.
pub trait FromSuperMatrix: Sized {
    /// Create an instance from a super matrix.
    fn from_super_matrix(_: &SuperMatrix) -> Option<Self>;
}

impl SuperMatrix {
    /// Create a matrix from a raw structure.
    ///
    /// The underlying memory is considered to be owned, and it will be freed
    /// when the object goes out of scope.
    pub unsafe fn from_raw(raw: ffi::SuperMatrix) -> SuperMatrix {
        SuperMatrix { raw,
                    rust_managed: false}
    }

    /// Consume the object returning the wrapped raw structure without freeing
    /// the underlying memory.
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

        let mut raw: ffi::SuperMatrix = unsafe {MaybeUninit::zeroed().assume_init()};


        let nzval: Vec<c_double> = mat.data().iter().map(|&x| x as c_double).collect();
        let rowind: Vec<c_int> = mat.indices().iter().map(|&x| x as c_int).collect();
        let mut colptr = Vec::new();
        let colptr_raw = mat.indptr();
        for ptr in colptr_raw.as_slice().unwrap() {
            colptr.push(ptr.clone() as c_int)
        }

        unsafe {ffi::dCreate_CompCol_Matrix(&mut raw, m, n, nnz, nzval.as_ptr() as *mut c_double,
                                            rowind.as_ptr() as *mut c_int, colptr.as_ptr() as *mut c_int,
                                            Stype_t::SLU_NC, Dtype_t::SLU_D, Mtype_t::SLU_GE);}

        unsafe { SuperMatrix::from_raw(raw) }
    }


    pub fn from_ndarray(array: Array2<f64>) -> Self {
        let nrows = array.nrows() as libc::c_int;
        let ncols = array.ncols() as libc::c_int;

        let col_major_data = unsafe { ffi::doubleMalloc(ncols * nrows) };
        let mut index: usize = 0;
        let col_major_data_ptr = unsafe { from_raw_parts_mut(col_major_data, (ncols * nrows) as usize) };
        for col in 0..ncols as usize {
            for row in 0..nrows as usize {
                col_major_data_ptr[index] = array[[row, col]];
                index += 1;
            }
        }

        let mut raw = MaybeUninit::<ffi::SuperMatrix>::uninit();

        unsafe {
            ffi::dCreate_Dense_Matrix(
                raw.as_mut_ptr(),
                nrows,
                ncols,
                col_major_data,
                nrows,
                ffi::Stype_t::SLU_DN,
                ffi::Dtype_t::SLU_D,
                ffi::Mtype_t::SLU_GE,
            );

            SuperMatrix { raw: raw.assume_init(),
                        rust_managed: true}
        }
    }

    pub fn into_ndarray(self) -> Option<Array2<f64>> {
        match self.raw.data_as_vec() {
            None => {None}
            Some(data) => {
                match Array2::from_shape_vec((self.nrows(), self.ncols()), data) {
                    Ok(arr) => { Some(arr.t().to_owned()) }
                    Err(_) => {None}
                }
            }
        }
    }

    pub fn nrows(&self) -> usize {
        self.raw.nrow as usize
    }

    pub fn ncols(&self) -> usize {
        self.raw.ncol as usize
    }

    pub fn raw(&self) -> &ffi::SuperMatrix {&self.raw}
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
                ffi::Stype_t::SLU_NC => unsafe {
                    ffi::Destroy_CompCol_Matrix(&mut self.raw);
                },
                ffi::Stype_t::SLU_NCP => unsafe {
                    ffi::Destroy_CompCol_Permuted(&mut self.raw);
                },
                ffi::Stype_t::SLU_NR => unsafe {
                    ffi::Destroy_CompRow_Matrix(&mut self.raw);
                },
                ffi::Stype_t::SLU_SC | ffi::Stype_t::SLU_SCP | ffi::Stype_t::SLU_SR => unsafe {
                    ffi::Destroy_SuperNode_Matrix(&mut self.raw);
                },
                ffi::Stype_t::SLU_DN => unsafe {
                    ffi::Destroy_Dense_Matrix(&mut self.raw);
                },
                _ => {},
            }
        }

    }
}

impl FromSuperMatrix for Compressed<f64> {
    fn from_super_matrix(matrix: &SuperMatrix) -> Option<Compressed<f64>> {
        use matrix::format::compressed::Variant;

        let raw = &matrix.raw;

        let rows = raw.nrow as usize;
        let columns = raw.ncol as usize;

        match (raw.Stype, raw.Dtype, raw.Mtype) {
            (ffi::Stype_t::SLU_NC, ffi::Dtype_t::SLU_D, ffi::Mtype_t::SLU_GE) => unsafe {
                let store = &*(raw.Store as *const ffi::NCformat);
                let nonzeros = store.nnz as usize;

                let mut values = Vec::with_capacity(nonzeros);
                let mut indices = Vec::with_capacity(nonzeros);
                let mut offsets = Vec::with_capacity(columns + 1);

                for i in 0..nonzeros {
                    values.push(*(store.nzval as *const libc::c_double).offset(i as isize));
                    indices.push(*store.rowind.offset(i as isize) as usize);
                }
                for i in 0..(columns + 1) {
                    offsets.push(*store.colptr.offset(i as isize) as usize);
                }

                Some(Compressed {
                    rows: rows,
                    columns: columns,
                    nonzeros: nonzeros,
                    variant: Variant::Column,
                    values: values,
                    indices: indices,
                    offsets: offsets,
                })
            },
            (ffi::Stype_t::SLU_NC, ffi::Dtype_t::SLU_D, _) => unimplemented!(),
            (ffi::Stype_t::SLU_NCP, ffi::Dtype_t::SLU_D, _) => unimplemented!(),
            _ => return None,
        }
    }
}
