#include "converter.h"

BackEnd::FramePtr Converter::Convert(FeatureTracker::FramePtr feat_frame, CameraPtr cam_master,
                                     StereoMatcher::FramePtr stereo_frame, CameraPtr cam_slave,
                                     const Eigen::VecVector3d& v_gyr, const Eigen::VecVector3d& v_acc,
                                     const std::vector<double>& v_imu_timestamp)
{
    BackEnd::FramePtr backend_frame(new BackEnd::Frame);
    backend_frame->timestamp = feat_frame->timestamp;
    backend_frame->pt_id = feat_frame->pt_id;
    for(int i = 0, n = feat_frame->pt.size(); i < n; ++i) {
        Eigen::Vector3d Pl, Pr;
        backend_frame->pt.emplace_back(feat_frame->pt[i].x, feat_frame->pt[i].y);
        Pl = cam_master->BackProject(backend_frame->pt[i]);
        backend_frame->pt_normal_plane.emplace_back(Pl);
        backend_frame->pt_r.emplace_back(stereo_frame->pt_r[i].x, stereo_frame->pt_r[i].y);
        if(stereo_frame->pt_r[i].x == -1) {
            backend_frame->pt_r_normal_plane.emplace_back(-1, -1, -1);
        }
        else {
            Pr = cam_slave->BackProject(backend_frame->pt_r[i]);
            backend_frame->pt_r_normal_plane.emplace_back(Pr);
        }
    }

    backend_frame->v_gyr = v_gyr;
    backend_frame->v_acc = v_acc;
    backend_frame->v_imu_timestamp = v_imu_timestamp;
    return backend_frame;
}

Relocalization::FramePtr Converter::Convert(FeatureTracker::FramePtr feat_frame,
                                            BackEnd::FramePtr back_frame,
                                            const Eigen::VecVector3d& v_x3Dw) {
    Relocalization::FramePtr frame(new Relocalization::Frame);
    frame->img = feat_frame->img;
    return frame;
}
