#include "frame.h"

uint64_t Frame::gNextFrameID = 0;
uint64_t Frame::gNextKeyFrameID = 0;

Frame::Frame(const cv::Mat& img_l, const cv::Mat& img_r)
    : mImgL(img_l), mImgR(img_r)
{
    mFrameID = gNextFrameID++;
}

Frame::~Frame() {}

void Frame::SetToKeyFrame() {
    mIsKeyFrame = true;
    mKeyFrameID = gNextKeyFrameID++;

    for(int i = 0, n = mv_uv.size(); i < n; ++i) {
        mvLastKFuv[i] = mv_uv[i];
    }
    std::cout << "set to keyframe " << mKeyFrameID << std::endl;
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

    std::cout << "frame" << mFrameID << " parallax_ave: " << parallax_ave << std::endl;

    if(parallax_ave > 10) {
        return true;
    }
    return  false;
}
