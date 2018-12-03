#include "frame.h"
#include "tracer.h"
#include "util_datatype.h"
#include "front_end/utility.h"
uint64_t Frame::gNextFrameID = 0;
uint64_t Frame::gNextKeyFrameID = 0;

Frame::Frame(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp)
    : mFrameID(gNextFrameID++), mIsKeyFrame(false), mKeyFrameID(0),
      mImgL(img_l), mImgR(img_r), mNumStereo(0), mTimeStamp(timestamp)
{
    Tracer::TraceBegin("lk_pyr");
    cv::buildOpticalFlowPyramid(mImgL, mImgPyrGradL, cv::Size(21, 21), 3);
    cv::buildOpticalFlowPyramid(mImgR, mImgPyrGradR, cv::Size(21, 21), 3);
    Tracer::TraceEnd();
    Tracer::TraceBegin("pyr");
    mImgPyrL = Utility::Pyramid(mImgL, 3);
    mImgPyrR = Utility::Pyramid(mImgR, 3);
    Tracer::TraceEnd();
}

Frame::~Frame() {}

void Frame::SetToKeyFrame() {
    mIsKeyFrame = true;
    mKeyFrameID = gNextKeyFrameID++;

    for(int i = 0, n = mvMapPoint.size(); i < n; ++i) {
        mvMapPoint[i]->AddMeas(shared_from_this(), i);
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

void Frame::SparseStereoMatching(double bf) {
    ScopedTrace st("SM");
    int N = mv_uv.size();
    int ur_size = mv_ur.size();

    if(N == ur_size)
        return;

    mv_ur.resize(N);
    for(int i = ur_size; i < N; ++i)
        mv_ur[i] = -1;

    uint32_t stereo_count = mNumStereo;

    std::vector<cv::Point2f> uv(mv_uv.begin() + ur_size, mv_uv.end());
    std::vector<cv::Point2f> pt_r;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(mImgPyrGradL, mImgPyrGradR, uv,
                             pt_r, status, err, cv::Size(21, 21), 3);

    const double MAX_DISPARITY = bf / 0.3;
    const double MIN_DISPARITY = 0;
    for(int i = 0, n = pt_r.size(); i < n; ++i) {
        if(status[i]) {
            if(InBorder(pt_r[i], mImgL.cols, mImgL.rows)) {
                double dy = std::abs(uv[i].y - pt_r[i].y);
                if(dy > 3)
                    continue;
                double dx = uv[i].x - pt_r[i].x;
                if(dx > MIN_DISPARITY && dx < MAX_DISPARITY) {
                    ++stereo_count;
                    mv_ur[i + ur_size] = pt_r[i].x;
                }
            }
        }
    }

    mNumStereo = stereo_count;
}
