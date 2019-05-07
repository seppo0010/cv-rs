//! Mat

use core::*;
use failure::Error;
use std::ffi::CString;
use std::mem;
use std::ops::{BitAnd, BitOr, BitXor, Not};
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::path::Path;
use std::slice;
use *;

/// The class `CMat` is used as a pointer to represent the Mat opencv structure
#[derive(Clone, Copy, Debug)]
pub enum CMat {}

impl CMat {
    pub(crate) fn new() -> *mut CMat {
        unsafe { cv_mat_new() }
    }
}

extern "C" {
    fn cv_mat_new() -> *mut CMat;
    fn cv_mat_from_file_storage(path: *const c_char, section: *const c_char) -> *mut CMat;
    fn cv_mat_new_with_size(rows: c_int, cols: c_int, t: c_int) -> *mut CMat;
    fn cv_mat_zeros(rows: c_int, cols: c_int, t: c_int) -> *mut CMat;
    fn cv_mat_from_buffer(rows: c_int, cols: c_int, t: CvType, buffer: *const u8) -> *mut CMat;
    fn cv_mat_is_valid(mat: *mut CMat) -> bool;
    fn cv_mat_rows(cmat: *const CMat) -> c_int;
    fn cv_mat_cols(cmat: *const CMat) -> c_int;
    fn cv_mat_depth(cmat: *const CMat) -> c_int;
    fn cv_mat_channels(cmat: *const CMat) -> c_int;
    fn cv_mat_data(cmat: *const CMat) -> *const u8;
    fn cv_mat_total(cmat: *const CMat) -> usize;
    fn cv_mat_step1(cmat: *const CMat, i: c_int) -> usize;
    fn cv_mat_elem_size(cmat: *const CMat) -> usize;
    fn cv_mat_elem_size1(cmat: *const CMat) -> usize;
    fn cv_mat_type(cmat: *const CMat) -> CvType;
    fn cv_mat_roi(cmat: *const CMat, rect: Rect) -> *mut CMat;
    fn cv_mat_flip(src: *mut CMat, code: c_int);
    fn cv_mat_drop(mat: *mut CMat);
    fn cv_mat_eye(rows: c_int, cols: c_int, cv_type: CvType) -> *mut CMat;
    fn cv_mat_in_range(cmat: *const CMat, lowerb: Scalar, upperb: Scalar, dst: *mut CMat);
    fn cv_mat_min_max_loc(
        cmat: *const CMat,
        min: *mut f64,
        max: *mut f64,
        min_loc: *mut Point2i,
        max_loc: *mut Point2i,
        cmask: *const CMat,
    );
    fn cv_mat_mix_channels(
        cmat: *const CMat,
        nsrcs: usize,
        dst: *mut CMat,
        ndsts: usize,
        from_to: *const c_int,
        npairs: usize,
    );
    fn cv_mat_normalize(csrc: *const CMat, cdst: *mut CMat, alpha: c_double, beta: c_double, norm_type: NormType);
    fn cv_mat_bitwise_and(src1: *const CMat, src2: *const CMat, dst: *mut CMat);
    fn cv_mat_bitwise_not(src: *const CMat, dst: *mut CMat);
    fn cv_mat_bitwise_or(src1: *const CMat, src2: *const CMat, dst: *mut CMat);
    fn cv_mat_bitwise_xor(src1: *const CMat, src2: *const CMat, dst: *mut CMat);
    fn cv_mat_merge(srcs: *const *mut CMat, srcs_len: c_ulong, dst: *mut CMat);
    fn cv_mat_count_non_zero(src: *const CMat) -> c_int;
    fn cv_mat_copy_make_border(
        src: *const CMat,
        dst: *mut CMat,
        top: c_int,
        bottom: c_int,
        left: c_int,
        right: c_int,
        border_type: c_int,
        color: Scalar,
    ) -> c_int;
    fn cv_mat_convert_to(src: *const CMat, cv_type: CvType, alpha: c_double, beta: c_double) -> *mut CMat;
    fn cv_mat_dft(src: *const CMat, dst: *mut CMat, flags: c_int, nonzero_rows: c_int);
    fn cv_mat_split(src: *const CMat, dst: *mut *mut CMat);
}

/// The class `Mat` represents an n-dimensional dense numerical single-channel or multi-channel array.
/// It can be used to store real or complex-valued vectors and matrices, grayscale or color images,
/// voxel volumes, vector fields, point clouds, tensors, histograms
#[derive(Debug)]
pub struct Mat {
    /// Pointer to the actual C/C++ data structure
    pub(crate) inner: *mut CMat,

    /// Number of columns
    pub cols: c_int,

    /// Number of rows
    pub rows: c_int,

    /// Depth of this mat (it should be the type).
    pub depth: c_int,

    /// Channels of this mat
    pub channels: c_int,
}

unsafe impl Send for CMat {}
unsafe impl Send for Mat {}
impl Into<CMat> for Mat {
    fn into(self) -> CMat {
        unsafe { *self.inner }
    }
}

impl Mat {
    /// Loads `Mat` from file storage
    pub fn from_file_storage<P: AsRef<Path>>(path: P, section: &str) -> Result<Mat, Error> {
        let path = path_to_cstring(path)?;
        let section = CString::new(section)?;

        let path = path.as_ptr();
        let section = section.as_ptr();
        let result = unsafe { cv_mat_from_file_storage(path, section) };
        Ok(Mat::from_raw(result))
    }

    #[inline]
    /// Creates a `Mat` object from raw `CMat` pointer. This will read the rows
    /// and cols of the image.
    pub(crate) fn from_raw(raw: *mut CMat) -> Mat {
        Mat {
            inner: raw,
            rows: unsafe { cv_mat_rows(raw) },
            cols: unsafe { cv_mat_cols(raw) },
            depth: unsafe { cv_mat_depth(raw) },
            channels: unsafe { cv_mat_channels(raw) },
        }
    }

    /// Creates an empty `Mat` struct.
    pub fn new() -> Mat {
        let m = CMat::new();
        Mat::from_raw(m)
    }

    /// Creates a new `Mat` from buffer. Note that internally opencv function
    /// won't take ownership of the Mat, but when we call `drop`, it will
    /// deallocate the memory. To prevent double-freeing, you must `mem::forget`
    /// it after use.
    ///
    /// The following example shows how to get the data from an image and create
    /// a new image with the data (also forgets it).
    ///
    /// ```rust,ignore
    /// let buffer = image.data();
    /// let size = image.size();
    /// let s = (size.width * size.height * 3) as usize;
    ///
    /// let mut vec = Vec::with_capacity(s);
    /// unsafe {
    ///   vec.set_len(s);
    ///   copy(buffer, vec.as_mut_ptr(), s);
    /// }
    /// let new_image = Mat::from_buffer(
    ///   size.height, size.width, CvType::Cv8UC3 as i32, &vec);
    ///
    ///  // . . . use new_image here, such as new_image.show(..) . . .
    ///
    /// ::std::mem::forget(new_image);
    /// ```
    pub fn from_buffer(rows: c_int, cols: c_int, cv_type: CvType, buf: &[u8]) -> Mat {
        let raw = unsafe { cv_mat_from_buffer(rows, cols, cv_type, buf.as_ptr()) };
        Mat::from_raw(raw)
    }

    /// Create an empty `Mat` with specific size (rows, cols and types).
    pub fn with_size(rows: c_int, cols: c_int, t: c_int) -> Self {
        let m = unsafe { cv_mat_new_with_size(rows, cols, t) };
        Mat::from_raw(m)
    }

    /// Create an empty `Mat` with specific size (rows, cols and types).
    pub fn zeros(rows: c_int, cols: c_int, t: c_int) -> Self {
        let m = unsafe { cv_mat_zeros(rows, cols, t) };
        Mat::from_raw(m)
    }

    /// Returns the raw data (as a `u8` pointer)
    pub fn data(&self) -> &[u8] {
        let bytes = unsafe { cv_mat_data(self.inner) };
        let len = self.total() * self.elem_size();
        unsafe { slice::from_raw_parts(bytes, len) }
    }

    /// Returns the total number of array elements. The method returns the
    /// number of array elements (a number of pixels if the array represents an
    /// image). For example, images with 1920x1080 resolution will return 2073600.
    pub fn total(&self) -> usize {
        unsafe { cv_mat_total(self.inner) }
    }

    /// Returns the matrix element size in bytes.
    ///
    /// The method returns the matrix element size in bytes. For example, if the
    /// matrix type is CV_16SC3 , the method returns 3*sizeof(short) or 6.
    pub fn elem_size(&self) -> usize {
        unsafe { cv_mat_elem_size(self.inner) }
    }

    /// Returns the size of each matrix element channel in bytes.
    ///
    /// The method returns the matrix element channel size in bytes, that
    /// is, it ignores the number of channels. For example, if the matrix
    /// type is CV_16SC3 , the method returns sizeof(short) or 2.
    pub fn elem_size1(&self) -> usize {
        unsafe { cv_mat_elem_size1(self.inner) }
    }

    /// Returns a normalized step.
    ///
    /// The method returns a matrix step divided by Mat::elemSize1() . It can be
    /// useful to quickly access an arbitrary matrix element
    pub fn step1(&self, i: c_int) -> usize {
        unsafe { cv_mat_step1(self.inner, i) }
    }

    /// Returns the size of this matrix.
    pub fn size(&self) -> Size2i {
        Size2i::new(self.cols, self.rows)
    }

    /// Check if the `Mat` is valid or not.
    pub fn is_valid(&self) -> bool {
        unsafe { cv_mat_is_valid(self.inner) }
    }

    /// Return a region of interest from a `Mat` specfied by a `Rect`.
    pub fn roi(&self, rect: Rect) -> Mat {
        let cmat = unsafe { cv_mat_roi(self.inner, rect) };
        Mat::from_raw(cmat)
    }

    /// Flips an image around vertical, horizontal, or both axes.
    pub fn flip(&mut self, code: FlipCode) {
        let code = match code {
            FlipCode::XAxis => 0,
            FlipCode::YAxis => 1,
            FlipCode::XYAxis => -1,
        };
        unsafe {
            cv_mat_flip(self.inner, code);
        }
    }

    /// Returns the images type. For supported types, please see
    /// [CvType](enum.CvType).
    pub fn cv_type(&self) -> CvType {
        unsafe { cv_mat_type(self.inner) }
    }

    /// Returns an identity matrix of the specified size and type.
    pub fn eye(rows: i32, cols: i32, cv_type: CvType) -> Mat {
        let result = unsafe { cv_mat_eye(rows, cols, cv_type) };
        Mat::from_raw(result)
    }

    /// Returns individual pixel (element) information within the Mat. This
    /// function may need type annotation to assist `FromBytes` trait.
    ///
    /// - If matrix is of type `CV_8U` then use `Mat.at<u8>(y,x)`.
    /// - If matrix is of type `CV_8S` then use `Mat.at<i8>(y,x)`.
    /// - If matrix is of type `CV_16U` then use `Mat.at<u16>(y,x)`.
    /// - If matrix is of type `CV_16S` then use `Mat.at<i16>(y,x)`.
    /// - If matrix is of type `CV_32S`  then use `Mat.at<i32>(y,x)`.
    /// - If matrix is of type `CV_32F`  then use `Mat.at<f32>(y,x)`.
    /// - If matrix is of type `CV_64F` then use `Mat.at<f64>(y,x)`.
    pub fn at<T: FromBytes>(&self, i0: i32) -> T {
        let data = self.data();
        let size = self.size();
        let pos = {
            if size.height == 1 {
                i0 as usize
            } else if size.width == 1 {
                i0 as usize * (self.step1(1) * self.elem_size1())
            } else {
                unimplemented! {};
            }
        };

        let byte = &data[pos];
        let ptr: *const _ = byte;
        let slice = unsafe { slice::from_raw_parts(ptr, mem::size_of::<T>()) };
        T::from_bytes(slice)
    }

    /// Returns individual pixel (element) information within the Mat. This
    /// function may need type annotation to assist `FromBytes` trait.
    ///
    /// See [Mat::at](struct.Mat.html#method.at) and
    /// [Mat::at3](struct.Mat.html#method.at3).
    pub fn at2<T: FromBytes>(&self, i0: i32, i1: i32) -> T {
        let data = self.data();
        let pos = i0 as usize * self.step1(0) * self.elem_size1() + i1 as usize * self.step1(1) * self.elem_size1();
        let byte = &data[pos];
        let ptr: *const _ = byte;
        let slice = unsafe { slice::from_raw_parts(ptr, mem::size_of::<T>()) };
        T::from_bytes(slice)
    }

    /// Returns individual pixel (element) information within the Mat. This
    /// function may need type annotation to assist `FromBytes` trait.
    ///
    /// See [Mat::at](struct.Mat.html#method.at) and
    /// [Mat::at2](struct.Mat.html#method.at2).
    pub fn at3<T: FromBytes>(&self, i0: i32, i1: i32, i2: i32) -> T {
        let data = self.data();
        let pos = i0 as usize * self.step1(0) * self.elem_size1()
            + i1 as usize * self.step1(1) * self.elem_size1()
            + i2 as usize;
        let byte = &data[pos];
        let ptr: *const _ = byte;
        let slice = unsafe { slice::from_raw_parts(ptr, mem::size_of::<T>()) };
        T::from_bytes(slice)
    }

    /// Checks if Mat elements lie between the elements of two other arrays
    /// (lowerb and upperb). The output Mat has the same size as `self` and
    /// CV_8U type.
    pub fn in_range(&self, lowerb: Scalar, upperb: Scalar) -> Mat {
        let m = CMat::new();
        unsafe { cv_mat_in_range(self.inner, lowerb, upperb, m) }
        Mat::from_raw(m)
    }

    /// Finds the global minimum and maximum in an array.
    ///
    /// This function finds the minimum and maximum element values and their
    /// positions. The extremums are searched across the whole array or, if mask
    /// is not an empty array, in the specified array region.
    ///
    /// N.B. Only work with single-channel Mat. For multi-channel arrays. If you
    /// need to find minimum or maximum elements across all the channels, use
    /// Mat::reshape first to reinterpret the array as single-channel. Or you
    /// may extract the particular channel using either extractImageCOI , or
    /// mixChannels, or split.
    pub fn min_max_loc(&self, mask: &Mat) -> (f64, f64, Point2i, Point2i) {
        let mut min = 0.0;
        let mut max = 0.0;
        let mut min_loc = Point2i::new(0, 0);
        let mut max_loc = Point2i::new(0, 0);
        unsafe { cv_mat_min_max_loc(self.inner, &mut min, &mut max, &mut min_loc, &mut max_loc, mask.inner) }
        (min, max, min_loc, max_loc)
    }

    /// Copy specified channels from `self` to the specified channels of output
    /// `Mat`.
    // The usage (self.depth) here is buggy, it should actually be the type!
    pub fn mix_channels<T: AsRef<[(c_int, c_int)]>>(&self, nsrcs: usize, ndsts: usize, from_to: T) -> Mat {
        let m = Mat::with_size(self.rows, self.cols, self.depth);
        let slice = from_to.as_ref();
        let ptr = slice.as_ptr() as *const c_int;
        unsafe {
            cv_mat_mix_channels(self.inner, nsrcs, m.inner, ndsts, ptr, slice.len());
        }
        m
    }

    /// Normalize the Mat according to the normalization type.
    pub fn normalize(&self, alpha: f64, beta: f64, t: NormType) -> Mat {
        let m = CMat::new();
        unsafe { cv_mat_normalize(self.inner, m, alpha, beta, t) }
        Mat::from_raw(m)
    }

    /// Counts non-zero array elements.
    pub fn count_non_zero(&self) -> c_int {
        unsafe { cv_mat_count_non_zero(self.inner) }
    }

    /// Forms a border around an image.
    ///
    /// The function copies the source image into the middle of the destination
    /// image. The areas to the left, to the right, above and below the copied
    /// source image will be filled with extrapolated pixels. This is not what
    /// filtering functions based on it do (they extrapolate pixels on-fly), but
    /// what other more complex functions, including your own, may do to
    /// simplify image boundary handling.
    pub fn copy_make_border(
        &self,
        top: i32,
        bottom: i32,
        left: i32,
        right: i32,
        type_: BorderType,
        color: Scalar,
    ) -> Mat {
        let m = CMat::new();
        unsafe {
            cv_mat_copy_make_border(self.inner, m, top, bottom, left, right, type_ as i32, color);
        }
        Mat::from_raw(m)
    }

    /// Changes mat type
    pub fn convert_to(&self, cv_type: CvType, alpha: f64, beta: f64) -> Mat {
        let m = unsafe { cv_mat_convert_to(self.inner, cv_type, alpha, beta) };
        Mat::from_raw(m)
    }

    /// Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array.
    pub fn dft(&self, flags: DFTFlag, nonzero_rows: i32) -> Mat {
        let m = CMat::new();
        unsafe {
            cv_mat_dft(self.inner, m, Into::<c_int>::into(flags), nonzero_rows);
        }
        Mat::from_raw(m)
    }

    /// Splits a mat into its channels
    pub fn split(&self) -> Vec<Mat> {
        let mut v: Vec<*mut CMat> = (0..self.channels).map(|_| CMat::new()).collect();
        unsafe { cv_mat_split(self.inner, v.as_mut_ptr()); }
        v.into_iter().map(|m| Mat::from_raw(m)).collect()
    }
}

/// Various border types, image boundaries are denoted with `|`.
#[derive(Debug, Copy, Clone)]
pub enum BorderType {
    /// `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    Constant = 0,
    /// `aaaaaa|abcdefgh|hhhhhhh`
    Replicate = 1,
    /// `fedcba|abcdefgh|hgfedcb`
    Reflect = 2,
    /// `cdefgh|abcdefgh|abcdefg`
    Wrap = 3,
    /// `gfedcb|abcdefgh|gfedcba`
    Reflect101 = 4,
    /// `uvwxyz|abcdefgh|ijklmno`
    Transparent = 5,
    /// Do not look outside of ROI.
    Isolated = 16,
}

impl BorderType {
    #[allow(non_upper_case_globals)]
    /// same as Reflect101
    pub const Default: BorderType = BorderType::Reflect101;
}

impl Drop for Mat {
    fn drop(&mut self) {
        unsafe {
            cv_mat_drop(self.inner);
        }
    }
}

impl BitAnd for Mat {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        let m = CMat::new();
        unsafe { cv_mat_bitwise_and(self.inner, rhs.inner, m) }
        Self::from_raw(m)
    }
}

impl<'a> BitAnd for &'a Mat {
    type Output = Mat;
    fn bitand(self, rhs: &'a Mat) -> Self::Output {
        let m = CMat::new();
        unsafe { cv_mat_bitwise_and(self.inner, rhs.inner, m) }
        Mat::from_raw(m)
    }
}

impl BitOr for Mat {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        let m = CMat::new();
        unsafe { cv_mat_bitwise_or(self.inner, rhs.inner, m) }
        Mat::from_raw(m)
    }
}

impl<'a> BitOr for &'a Mat {
    type Output = Mat;
    fn bitor(self, rhs: &'a Mat) -> Self::Output {
        let m = CMat::new();
        unsafe { cv_mat_bitwise_or(self.inner, rhs.inner, m) }
        Mat::from_raw(m)
    }
}

impl BitXor for Mat {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let m = CMat::new();
        unsafe { cv_mat_bitwise_xor(self.inner, rhs.inner, m) }
        Mat::from_raw(m)
    }
}

impl<'a> BitXor for &'a Mat {
    type Output = Mat;
    fn bitxor(self, rhs: &'a Mat) -> Self::Output {
        let m = CMat::new();
        unsafe { cv_mat_bitwise_xor(self.inner, rhs.inner, m) }
        Mat::from_raw(m)
    }
}

impl Not for Mat {
    type Output = Self;
    fn not(self) -> Self::Output {
        let m = CMat::new();
        unsafe { cv_mat_bitwise_not(self.inner, m) }
        Mat::from_raw(m)
    }
}

impl Clone for Mat {
    fn clone(&self) -> Self {
        Mat::from_buffer(self.rows, self.cols, self.cv_type(), self.data())
    }
}

impl<'a> Not for &'a Mat {
    type Output = Mat;
    fn not(self) -> Self::Output {
        let m = CMat::new();
        unsafe { cv_mat_bitwise_not(self.inner, m) }
        Mat::from_raw(m)
    }
}

/// Merge trait to merge mats into a multi-channel array.
pub trait Merge {
    /// The functions merge several arrays to make a single multi-channel array.
    fn merge(&self) -> Mat;
}

impl Merge for &[Mat] {
    /// The functions merge several arrays to make a single multi-channel array. That is,
    /// each element of the output array will be a concatenation of the elements of the input
    /// arrays, where elements of i-th input array are treated as mv[i].channels()-element vectors.
    ///
    /// https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=merge#cv2.merge
    fn merge(&self) -> Mat {
        let me = self.iter().collect::<Vec<&Mat>>();
        me.merge()
    }
}

impl Merge for Vec<Mat> {
    /// The functions merge several arrays to make a single multi-channel array. That is,
    /// each element of the output array will be a concatenation of the elements of the input
    /// arrays, where elements of i-th input array are treated as mv[i].channels()-element vectors.
    ///
    /// https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=merge#cv2.merge
    fn merge(&self) -> Mat {
        let me = self.iter().collect::<Vec<&Mat>>();
        me.merge()
    }
}

impl Merge for &[&Mat] {
    /// The functions merge several arrays to make a single multi-channel array. That is,
    /// each element of the output array will be a concatenation of the elements of the input
    /// arrays, where elements of i-th input array are treated as mv[i].channels()-element vectors.
    ///
    /// https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=merge#cv2.merge
    fn merge(&self) -> Mat {
        let m = CMat::new();
        let myinners: Vec<*mut CMat> = self.iter().map(|x| x.inner).collect::<Vec<_>>();
        unsafe { cv_mat_merge(myinners.as_ptr(), self.len() as c_ulong, m); }
        Mat::from_raw(m)
    }
}

impl Merge for Vec<&Mat> {
    /// The functions merge several arrays to make a single multi-channel array. That is,
    /// each element of the output array will be a concatenation of the elements of the input
    /// arrays, where elements of i-th input array are treated as mv[i].channels()-element vectors.
    ///
    /// https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=merge#cv2.merge
    fn merge(&self) -> Mat {
        let me: &[_] = self;
        me.merge()
    }
}

/// Flags for Discrete Fourier transform
#[derive(Debug, Clone)]
pub enum DFTFlag {
    /// No flag
    None,
    /// Performs an inverse 1D or 2D transform instead of the default forward transform
    Inverse,
    /// scales the result: divide it by the number of array elements. Normally, it is combined with Inverse
    Scale,
    /// performs a forward or inverse transform of every individual row of the input matrix
    Rows,
    /// performs a forward transformation of 1D or 2D real array; the result, though being a
    /// complex array, has complex-conjugate symmetry (CCS, see the function description below for
    /// details), and such an array can be packed into a real array of the same size as input,
    /// which is the fastest option and which is what the function does by default; however, you
    /// may wish to get a full complex array (for simpler spectrum analysis, and so on) - pass the
    /// flag to enable the function to produce a full-size complex output array.
    ComplexOutput,
    /// performs an inverse transformation of a 1D or 2D complex array; the result is normally a
    /// complex array of the same size, however, if the input array has conjugate-complex symmetry
    /// (for example, it is a result of forward transformation with DFT_COMPLEX_OUTPUT flag), the
    /// output is a real array; while the function itself does not check whether the input is
    /// symmetrical or not, you can pass the flag and then the function will assume the symmetry
    /// and produce the real output array (note that when the input is packed into a real array and
    /// inverse transformation is executed, the function treats the input as a packed
    /// complex-conjugate symmetrical array, and the output will also be a real array).
    RealOutput,
    /// A combination of the other flags
    Multiple(Vec<DFTFlag>),
}

impl Into<c_int> for DFTFlag {
    fn into(self) -> c_int {
        match self {
            DFTFlag::None => 0,
            DFTFlag::Inverse => 1,
            DFTFlag::Scale => 2,
            DFTFlag::Rows => 4,
            DFTFlag::ComplexOutput => 16,
            DFTFlag::RealOutput => 32,
            DFTFlag::Multiple(v) => v.into_iter().fold(0, |acc, x| acc | Into::<i32>::into(x)),
        }
    }
}

impl BitOr for DFTFlag {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        let mut flags = Vec::new();
        match self {
            crate::DFTFlag::Multiple(ref x) => flags.extend(x.iter().cloned()),
            v => flags.push(v),
        }
        match rhs {
            crate::DFTFlag::Multiple(ref x) => flags.extend(x.iter().cloned()),
            v => flags.push(v),
        }
        DFTFlag::Multiple(flags)
    }
}
