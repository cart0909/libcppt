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

    void RemoveOutlierFromF();
    void CheckStereoConstrain();

    void SetMask();

    cv::Mat mMask;
//    bool mbFirstImage = true;
    cv::Mat mCurImage, mCurImageR;
    cv::Mat mLastImage, mLastImageR;
    std::vector<cv::Point2f> mvNewPts;

    // size all same
    std::vector<cv::Point2f> mvLastPts, mvCurPts;
    std::vector<cv::Point2f> mvLastUnPts, mvCurUnPts;
    std::vector<cv::Point2f> mvCurPtsR;
    int mNumStereo;
    std::vector<uchar> mvIsStereo;
    std::vector<int> mvTrackCnt;
    std::vector<uint64_t> mvIds;
    std::vector<double> mvInvDepth;
    uint64_t mNextPtId;

    StereoCameraPtr mpStereoCam;
};

using ImageProcessorPtr = std::shared_ptr<ImageProcessor>;
using ImageProcessorConstPtr = std::shared_ptr<const ImageProcessor>;
