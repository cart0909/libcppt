#include "simple_frontend.h"
#include "basic_datatype/tic_toc.h"
#include "tracer.h"

template<class T>
void ReduceVector(std::vector<T> &v, const std::vector<uchar>& status) {
    int j = 0;
    for(int i = 0, n = v.size(); i < n; ++i) {
        if(status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}

bool InBorder(const cv::Point2f& pt, int width, int height) {
    const int border_size = 1;
    int img_x = std::round(pt.x);
    int img_y = std::round(pt.y);
    if(img_x - border_size < 0 || img_y - border_size <= 0 || img_x + border_size >= width ||
            img_y + border_size >= height)
        return false;
    return true;
}

SimpleFrontEnd::SimpleFrontEnd(const SimpleStereoCamPtr& camera)
    : mpCamera(camera), mFeatureID(0)
{

}

SimpleFrontEnd::~SimpleFrontEnd() {

}

// with frame without any feature extract by FAST corner detector
// if exist some features extract FAST corner in empty grid.
void SimpleFrontEnd::ExtractFeatures(FramePtr frame) {
    ScopedTrace st("EFeat");
    // uniform the feature distribution
    UniformFeatureDistribution(frame);

    static int empty_value = -1, exist_value = -2;
    int grid_rows = std::ceil(static_cast<float>(mpCamera->height) / 32);
    int grid_cols = std::ceil(static_cast<float>(mpCamera->width) / 32);
    std::vector<std::vector<int>> grids(grid_rows, std::vector<int>(grid_cols, empty_value));
    for(int i = 0, n = frame->mv_uv.size(); i < n; ++i) {
        auto& pt = frame->mv_uv[i];
        int grid_i = pt.y / 32;
        int grid_j = pt.x / 32;
        if(grids[grid_i][grid_j] != exist_value)
            grids[grid_i][grid_j] = exist_value;
    }

    std::vector<cv::KeyPoint> kps;
    Tracer::TraceBegin("FAST");
    cv::FAST(frame->mImgL, kps, 20);
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
                frame->mvPtID.emplace_back(mFeatureID++);
                frame->mvPtCount.emplace_back(0);
                frame->mv_uv.emplace_back(kps[j].pt);
                frame->mvLastKFuv.emplace_back(kps[j].pt);
            }
}

// track features by optical flow and check epipolar constrain
void SimpleFrontEnd::TrackFeaturesByOpticalFlow(FramePtr ref_frame, FramePtr cur_frame) {
    if(ref_frame->mv_uv.empty())
        return;
    ScopedTrace st("TrackFeat");

    const int width = mpCamera->width, height = mpCamera->height;

    // copy ref frame data for cur frame
    std::vector<cv::Point2f> ref_frame_pts = ref_frame->mv_uv;
    cur_frame->mvPtID = ref_frame->mvPtID;
    cur_frame->mvPtCount = ref_frame->mvPtCount;
    cur_frame->mvLastKFuv = ref_frame->mvLastKFuv;

    // optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    Tracer::TraceBegin("Optical Flow");
    cv::calcOpticalFlowPyrLK(ref_frame->mImgL, cur_frame->mImgL, ref_frame_pts,
                             cur_frame->mv_uv, status, err, cv::Size(21, 21), 3);
    Tracer::TraceEnd();

    for(int i = 0, n = cur_frame->mv_uv.size(); i < n; ++i)
        if(status[i] && !InBorder(cur_frame->mv_uv[i], width, height))
            status[i] = 0;

    Tracer::TraceBegin("reduce vector");
    ReduceVector(ref_frame_pts, status);
    ReduceVector(cur_frame->mv_uv, status);
    ReduceVector(cur_frame->mvPtID, status);
    ReduceVector(cur_frame->mvPtCount, status);
    ReduceVector(cur_frame->mvLastKFuv, status);
    Tracer::TraceEnd();

    RemoveOutlierFromF(ref_frame_pts, cur_frame);

    for(auto& it : cur_frame->mvPtCount)
        ++it;
}

// simple sparse stereo matching algorithm by optical flow
void SimpleFrontEnd::SparseStereoMatching(FramePtr frame) {
    ScopedTrace st("SM");
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> pt_r;
    frame->mv_ur.resize(frame->mv_uv.size(), -1);
    // optical flow between left and right image
    cv::calcOpticalFlowPyrLK(frame->mImgL, frame->mImgR, frame->mv_uv,
                             pt_r, status, err, cv::Size(21, 21), 3);

    uint32_t stereo_count = 0;
    static const double MAX_DISPARITY = mpCamera->bf / 0.3;
    static const double MIN_DISPARITY = 0;
    for(int i = 0, n = pt_r.size(); i < n; ++i)
        if(status[i])
            if(InBorder(pt_r[i], mpCamera->width, mpCamera->height)) {
                double dy = std::abs(frame->mv_uv[i].y - pt_r[i].y);
                if(dy > 3)
                    continue;
                double dx = frame->mv_uv[i].x - pt_r[i].x;
                if(dx > MIN_DISPARITY && dx < MAX_DISPARITY) {
                    ++stereo_count;
                    frame->mv_ur[i] = pt_r[i].x;
                }
            }
}

void SimpleFrontEnd::RemoveOutlierFromF(std::vector<cv::Point2f>& ref_pts,
                                        FramePtr cur_frame) {
    if(ref_pts.size() > 8) {
        ScopedTrace st("RemoveF");
        std::vector<uchar> status;
        cv::findFundamentalMat(ref_pts, cur_frame->mv_uv, cv::FM_RANSAC, 1, 0.99, status);
        ReduceVector(ref_pts, status);
        ReduceVector(cur_frame->mv_uv, status);
        ReduceVector(cur_frame->mvPtID, status);
        ReduceVector(cur_frame->mvPtCount, status);
        ReduceVector(cur_frame->mvLastKFuv, status);
    }
}

void SimpleFrontEnd::UniformFeatureDistribution(FramePtr cur_frame) {
    ScopedTrace st("UDist");
    static int empty_value = -1;
    int grid_rows = std::ceil(static_cast<float>(mpCamera->height)/32);
    int grid_cols = std::ceil(static_cast<float>(mpCamera->width)/32);
    std::vector<std::vector<int>> grids(grid_rows, std::vector<int>(grid_cols, empty_value));
    for(int i = 0, n = cur_frame->mv_uv.size(); i < n; ++i) {
        auto& pt = cur_frame->mv_uv[i];
        int grid_i = pt.y / 32;
        int grid_j = pt.x / 32;
        int idx = grids[grid_i][grid_j];
        if(idx != empty_value) {
            if(cur_frame->mvPtCount[i] > cur_frame->mvPtCount[idx])
                grids[grid_i][grid_j] = i;
        }
        else
            grids[grid_i][grid_j] = i;
    }

    std::vector<uint64_t> temp_pt_id;
    std::vector<uint32_t> temp_pt_count;
    std::vector<cv::Point2f> temp_pts;
    std::vector<cv::Point2f> temp_pts_lastkf;
    for(auto& i : grids) {
        for(auto& j : i) {
            if(j != empty_value) {
                temp_pt_id.emplace_back(cur_frame->mvPtID[j]);
                temp_pt_count.emplace_back(cur_frame->mvPtCount[j]);
                temp_pts.emplace_back(cur_frame->mv_uv[j]);
                temp_pts_lastkf.emplace_back(cur_frame->mvLastKFuv[j]);
            }
        }
    }

    cur_frame->mvPtID = std::move(temp_pt_id);
    cur_frame->mvPtCount = std::move(temp_pt_count);
    cur_frame->mv_uv = std::move(temp_pts);
    cur_frame->mvLastKFuv = std::move(temp_pts_lastkf);
}
