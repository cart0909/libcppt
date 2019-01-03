#pragma once
#include <memory>
#include <vector>
#include <sophus/se3.hpp>
#include "util_datatype.h"
#include "mappoint.h"
#include "camera_model/simple_stereo_camera.h"
#include "front_end/imu_preintegration.h"

class MapPoint;
SMART_PTR(MapPoint)

class Frame : public std::enable_shared_from_this<Frame> {
public:
    Frame(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp,
          SimpleStereoCamPtr camera);
    ~Frame();

    void set_bad();
    bool CheckKeyFrame();
    std::vector<MapPointPtr> SetToKeyFrame();

    // simple sparse stereo matching algorithm by optical flow
    void SparseStereoMatching(double bf);
    void ExtractFAST();
    void ExtractGFTT();

    static uint64_t gNextFrameID, gNextKeyFrameID;
    uint64_t mFrameID;
    bool mIsKeyFrame;
    uint64_t mKeyFrameID;

    Sophus::SE3d mTwc;

    // image points
    std::vector<uint32_t> mvPtCount;
    std::vector<cv::Point2f> mv_uv;
    std::vector<cv::Point2f> mvLastKFuv;
    std::vector<float> mv_ur; // value -1 is mono point
    std::vector<MapPointPtr> mvMapPoint; // FIXME
    uint32_t mNumStereo;

    // image
    cv::Mat mImgL, mImgR;
    // CLAHE img for tracking
    cv::Mat mClaheL, mClaheR;
    // image pyr for optical flow
    ImagePyr mImgPyrGradL, mImgPyrGradR; // will remove when opencv LK remove
    ImagePyr mImgPyrL, mImgPyrR;
    double mTimeStamp;

    // ceres
    double vertex_data[16];

    SimpleStereoCamPtr mpCamera;

    // imu
    Eigen::Vector3d Pw;
    Sophus::SO3d Rwb;
    Eigen::Vector3d Vw;
    Eigen::Vector3d ba;
    Eigen::Vector3d bg;

    // integrate from last kf to this frame
    ImuPreintegrationPtr imu_preintegration;
};

SMART_PTR(Frame)
