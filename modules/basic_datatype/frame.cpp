#include "frame.h"
#include "tracer.h"
#include "util_datatype.h"
#include "front_end/utility.h"
uint64_t Frame::gNextFrameID = 0;
uint64_t Frame::gNextKeyFrameID = 0;

Frame::Frame(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp, SimpleStereoCamPtr camera)
    : mFrameID(gNextFrameID++), mIsKeyFrame(false), mKeyFrameID(0),
      mImgL(img_l), mImgR(img_r), mNumStereo(0), mTimeStamp(timestamp),
      mpCamera(camera)
{
    Tracer::TraceBegin("CLAHE");
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0);
    clahe->apply(mImgL, mClaheL);
    clahe->apply(mImgR, mClaheR);
    Tracer::TraceEnd();
    Tracer::TraceBegin("lk_pyr");
    cv::buildOpticalFlowPyramid(mClaheL, mImgPyrGradL, cv::Size(21, 21), 3);
    cv::buildOpticalFlowPyramid(mClaheR, mImgPyrGradR, cv::Size(21, 21), 3);
    Tracer::TraceEnd();
    Tracer::TraceBegin("pyr");
    mImgPyrL = Utility::Pyramid(mClaheL, 3);
    mImgPyrR = Utility::Pyramid(mClaheR, 3);
    Tracer::TraceEnd();
}

Frame::~Frame() {

}

void Frame::set_bad() {
    for(auto& mp : mvMapPoint) {
        if(mp->is_bad())
            continue;
        if(mp->get_parent().first == shared_from_this()) {
            mp->set_bad();
        }
    }
}

std::vector<MapPointPtr> Frame::SetToKeyFrame() {
    std::vector<MapPointPtr> temp;
    mIsKeyFrame = true;
    mKeyFrameID = gNextKeyFrameID++;

    for(int i = 0, n = mvMapPoint.size(); i < n; ++i) {
        if(mvMapPoint[i]->is_bad()) { // red point
            if(mvMapPoint[i]->is_init()) { // exist 3D
                Eigen::Vector3d x3Dc = mvMapPoint[i]->x3Dc(mTwc.inverse());
                mvMapPoint[i] = std::make_shared<MapPoint>();
                mvMapPoint[i]->add_meas(shared_from_this(), i);
                mvMapPoint[i]->inv_z(1.0 / x3Dc(2));
            }
            else { // only 2d
                mvMapPoint[i] = std::make_shared<MapPoint>();
                mvMapPoint[i]->add_meas(shared_from_this(), i);
            }
            temp.emplace_back(mvMapPoint[i]);
        }
        else // yellow or green
            mvMapPoint[i]->add_meas(shared_from_this(), i);
    }
    return temp;
}

bool Frame::CheckKeyFrame() {
    int num_features = mv_uv.size();
    if(num_features < 150) {
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

void Frame::ExtractFAST() {
    ScopedTrace st("ExtractFAST");
    // uniform the feature distribution
//    UniformFeatureDistribution(frame); // FIXME!!!

    static int empty_value = -1, exist_value = -2;
    int grid_rows = std::ceil(static_cast<float>(mImgL.rows) / 32);
    int grid_cols = std::ceil(static_cast<float>(mImgL.cols) / 32);
    std::vector<std::vector<int>> grids(grid_rows, std::vector<int>(grid_cols, empty_value));
    for(int i = 0, n = mv_uv.size(); i < n; ++i) {
        auto& pt = mv_uv[i];
        int grid_i = pt.y / 32;
        int grid_j = pt.x / 32;
        if(grids[grid_i][grid_j] != exist_value)
            grids[grid_i][grid_j] = exist_value;
    }

    std::vector<cv::KeyPoint> kps;
    Tracer::TraceBegin("FAST20");
    cv::FAST(mImgL, kps, 20);
    Tracer::TraceEnd();

    for(int i = 0, n = kps.size(); i < n; ++i) {
        auto& pt = kps[i].pt;
        int grid_i = pt.y / 32;
        int grid_j = pt.x / 32;
        if(grid_i >= grid_rows)
            grid_i = grid_rows - 1;
        if(grid_j >= grid_cols)
            grid_j = grid_cols - 1;
        int value = grids[grid_i][grid_j];
        if(value != exist_value) {
            if(value == empty_value) {
                grids[grid_i][grid_j] = i;
            }
            else {
                if(kps[i].response > kps[value].response)
                    grids[grid_i][grid_j] = i;
            }
        }
    }

    for(auto& i : grids)
        for(auto& j : i)
            if(j >= 0) {
                mvPtCount.emplace_back(0);
                mv_uv.emplace_back(kps[j].pt);
                mvLastKFuv.emplace_back(kps[j].pt);
                mvMapPoint.emplace_back(new MapPoint);
            }

    if(mv_uv.size() < 150) {
        grids.resize(grid_rows, std::vector<int>(grid_cols, empty_value));

        for(int i = 0, n = mv_uv.size(); i < n; ++i) {
            auto& pt = mv_uv[i];
            int grid_i = pt.y / 32;
            int grid_j = pt.x / 32;
            if(grids[grid_i][grid_j] != exist_value)
                grids[grid_i][grid_j] = exist_value;
        }

        Tracer::TraceBegin("FAST7");
        cv::FAST(mImgL, kps, 7);
        Tracer::TraceEnd();

        for(int i = 0, n = kps.size(); i < n; ++i) {
            auto& pt = kps[i].pt;
            int grid_i = pt.y / 32;
            int grid_j = pt.x / 32;
            if(grid_i >= grid_rows)
                grid_i = grid_rows - 1;
            if(grid_j >= grid_cols)
                grid_j = grid_cols - 1;
            int value = grids[grid_i][grid_j];
            if(value != exist_value) {
                if(value == empty_value) {
                    grids[grid_i][grid_j] = i;
                }
                else {
                    if(kps[i].response > kps[value].response)
                        grids[grid_i][grid_j] = i;
                }
            }
        }

        for(auto& i : grids)
            for(auto& j : i)
                if(j >= 0) {
                    mvPtCount.emplace_back(0);
                    mv_uv.emplace_back(kps[j].pt);
                    mvLastKFuv.emplace_back(kps[j].pt);
                    mvMapPoint.emplace_back(new MapPoint);
                }
    }
}

void Frame::ExtractGFTT() {
    ScopedTrace st("ExtractGFTT");
    cv::Mat mask = cv::Mat(mImgL.rows, mImgL.cols, CV_8U, cv::Scalar(255));

    for(auto& pt : mv_uv) {
        cv::circle(mask, pt, 20, cv::Scalar(0), -1);
    }

    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(mImgL, pts, 0, 0.01, 20, mask); // 4.954 ms

    for(auto& pt : pts) {
        mvPtCount.emplace_back(0);
        mv_uv.emplace_back(pt);
        mvLastKFuv.emplace_back(pt);
        mvMapPoint.emplace_back(new MapPoint);
    }
}
