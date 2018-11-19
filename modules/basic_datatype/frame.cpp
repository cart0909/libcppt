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
}
