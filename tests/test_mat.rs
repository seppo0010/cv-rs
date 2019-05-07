extern crate cv;
mod utils;

use cv::*;

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
