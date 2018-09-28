#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "frontend_config.h"
#include "basic_datatype/imu_data.h"
#include "camera_model/stereo_camera.h"

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();

    void ReadStereo(const cv::Mat& imLeft, const cv::Mat& imRight, double timestamp);
    void ReadStereo(const cv::Mat& imLeft, const cv::Mat& imRight, double timestamp,
                    const std::vector<ImuData>& imu_data);

    void SetMask();

    cv::Mat mMask;
//    bool mbFirstImage = true;
    std::vector<cv::Mat> mvCurImagePyramid, mvCurImagePyramidR;
    std::vector<cv::Mat> mvLastImagePyramid, mvLastImagePyramidR;
    std::vector<cv::Point2f> mvNewPts;

    // size all same
    std::vector<cv::Point2f> mvLastPts, mvCurPts;
    std::vector<cv::Point2f> mvLastPtsR, mvCurPtsR;
    std::vector<int> mvTrackCnt;

    StereoCameraPtr mpStereoCam;
};

using ImageProcessorPtr = std::shared_ptr<ImageProcessor>;
using ImageProcessorConstPtr = std::shared_ptr<const ImageProcessor>;
