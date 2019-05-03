extern crate cv;
mod utils;

use cv::*;

#[test]
fn test_dft_flag() {
    assert_eq!(
        Into::<i32>::into(DFTFlag::Inverse | DFTFlag::Inverse | DFTFlag::RealOutput),
        33,
    )
}

#[test]
fn test_dft() {
    // steps from https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
    let img = utils::load_lenna();
    let m = get_optimal_dft_size(img.size().height);
    let n = get_optimal_dft_size(img.size().width);
    let padded = img.copy_make_border(0, m - img.rows, 0, n - img.cols, BorderType::Constant, Scalar::all(0));
    let complex_i = vec![padded, Mat::zeros(img.size().height, img.size().width, img.cv_type() as i32)].merge();
    let _ = complex_i.dft(DFTFlag::None, 0);
}

