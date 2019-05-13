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
    let img = utils::load_lenna().convert_to(cv::CvType::Cv32FC1, 1.0, 0.0);
    let m = get_optimal_dft_size(img.size().height);
    let n = get_optimal_dft_size(img.size().width);
    let padded = img.copy_make_border(0, m - img.rows, 0, n - img.cols, BorderType::Constant, Scalar::all(0));
    let complex_i = vec![padded, Mat::zeros(img.size().height, img.size().width, cv::CvType::Cv32FC1 as i32)].merge();
    let _ = complex_i.dft(DFTFlag::None, 0);
}

#[test]
fn test_split() {
    let mat = Mat::from_buffer(3, 1, cv::core::CvType::Cv8UC2, &[1, 2, 3, 4, 5, 6]);
    assert_eq!(mat.channels, 2);
    let split = mat.split();
    assert_eq!(split[0].data(), &[1, 3, 5]);
    assert_eq!(split[0].channels, 1);
    assert_eq!(split[0].cv_type(),  cv::core::CvType::Cv8UC1);
    assert_eq!(split[1].data(), &[2, 4, 6]);
    assert_eq!(split[1].channels, 1);
    assert_eq!(split[1].cv_type(),  cv::core::CvType::Cv8UC1);
}

#[test]
fn test_magnitude() {
    let mat = Mat::magnitude(
        &Mat::from_buffer(3, 1, cv::core::CvType::Cv8UC1, &[1, 2, 3]).convert_to(cv::CvType::Cv32FC1, 1.0, 0.0),
        &Mat::from_buffer(3, 1, cv::core::CvType::Cv8UC1, &[4, 5, 6]).convert_to(cv::CvType::Cv32FC1, 1.0, 0.0),
    );
    assert_eq!(mat.size().width, 1);
    assert_eq!(mat.size().height, 3);
    assert_eq!(mat.convert_to(cv::CvType::Cv8UC1, 1.0, 0.0).data(), &[4, 5, 7]);
}

#[test]
fn test_pow() {
    let mat = Mat::from_buffer(3, 1, cv::core::CvType::Cv8UC2, &[1, 2, 3, 4, 5, 6]);
    let powered = mat.pow(2.0);
    assert_eq!(powered.data(), &[1, 4, 9, 16, 25, 36]);
}

#[test]
fn test_divide() {
    let mat1 = Mat::from_buffer(3, 1, cv::core::CvType::Cv8UC2, &[10, 20, 30, 80, 100, 6]);
    let mat2 = Mat::from_buffer(3, 1, cv::core::CvType::Cv8UC2, &[1, 2, 3, 4, 5, 6]);
    let divided = mat1.divide(&mat2, 1.0, -1);
    assert_eq!(divided.data(), &[10, 10, 10, 20, 20, 1]);
}
