#include "frame.h"
#include "tracer.h"
uint64_t Frame::gNextFrameID = 0;
uint64_t Frame::gNextKeyFrameID = 0;

Frame::Frame(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp)
    : mFrameID(gNextFrameID++), mIsKeyFrame(false), mKeyFrameID(0),
      mImgL(img_l), mImgR(img_r), mNumStereo(0), mTimeStamp(timestamp)
{
    ScopedTrace st("build_pyr");
    cv::buildOpticalFlowPyramid(mImgL, mImgPyrL, cv::Size(21, 21), 3);
    cv::buildOpticalFlowPyramid(mImgR, mImgPyrR, cv::Size(21, 21), 3);
}

Frame::~Frame() {}

void Frame::SetToKeyFrame() {
    mIsKeyFrame = true;
    mKeyFrameID = gNextKeyFrameID++;

    for(int i = 0, n = mvMapPoint.size(); i < n; ++i) {
        mvMapPoint[i]->AddMeas(shared_from_this(), mv_uv[i]);
    }
}

bool Frame::CheckKeyFrame() {
    int num_features = mv_uv.size();
    if(num_features < 100) {
        return true;
    }

    float parallax_sum = 0;
    for(int i = 0, n = mv_uv.size(); i < n; ++i) {
        float dx = mv_uv[i].x - mvLastKFuv[i].x;
        float dy = mv_uv[i].y - mvLastKFuv[i].y;
        parallax_sum += std::sqrt(dx*dx + dy*dy);
    }
    float parallax_ave = parallax_sum / num_features;

    if(parallax_ave > 10) {
        return true;
    }
    return  false;
}
