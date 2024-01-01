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
use std::slice::from_raw_parts_mut;
use sprs::CsMat;

/// A super matrix.
pub struct SuperMatrix {
    raw: ffi::SuperMatrix,
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
        SuperMatrix { raw: raw }
    }

    /// Consume the object returning the wrapped raw structure without freeing
    /// the underlying memory.
    pub fn into_raw(self) -> ffi::SuperMatrix {
        let raw = self.raw;
        mem::forget(self);
        raw
    }

    pub fn from_csc_mat(mat: CsMat<f64>) -> Self {
        assert_eq!(mat.storage(), sprs::CompressedStorage::CSC);

        let nrows = mat.rows();
        let ncols = mat.cols();
        let nnz = mat.nnz();

        // Bind the indptr to a variable
        let indptr_binding = mat.indptr();
        let indptr = indptr_binding.as_slice();

        let indices = mat.indices();
        let data = mat.data();

        let raw = ffi::SuperMatrix {
            Stype: ffi::Stype_t::SLU_NC,
            Dtype: ffi::Dtype_t::SLU_D,
            Mtype: ffi::Mtype_t::SLU_GE,
            nrow: nrows as libc::c_int,
            ncol: ncols as libc::c_int,
            Store: Box::into_raw(Box::new(ffi::NCformat {
                nnz: nnz as libc::c_int,
                nzval: data.as_ptr() as *mut libc::c_void,
                rowind: indices.as_ptr() as *mut libc::c_int,
                colptr: indptr.unwrap().as_ptr() as *mut libc::c_int,
            })) as *mut libc::c_void,
        };

        // Prevent Rust from dropping the original vectors
        mem::forget(mat);
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

            mem::forget(col_major_data_ptr);

            SuperMatrix { raw: raw.assume_init() }
        }
    }

    pub fn into_ndarray(self) -> Option<ndarray::Array2<f64>> {
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
