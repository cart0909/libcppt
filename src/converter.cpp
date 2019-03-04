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
            backend_frame->pt_r_normal_plane.emplace_back(-1, -1, 0);
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
                                            const Eigen::VecVector3d& v_x3Dc) {
    Relocalization::FramePtr frame(new Relocalization::Frame);
    frame->img = feat_frame->img;
    frame->compressed_img = feat_frame->compressed_img;
    frame->timestamp = feat_frame->timestamp;
    for(int i = 0, n = v_x3Dc.size(); i < n; ++i) {
        if(v_x3Dc[i](2) > 0) {
            frame->v_pt_id.emplace_back(feat_frame->pt_id[i]);
            frame->v_pt_2d_uv.emplace_back(feat_frame->pt[i]);
            frame->v_pt_2d_normal.emplace_back(back_frame->pt_normal_plane[i](0),
                                               back_frame->pt_normal_plane[i](1));
            frame->v_pt_3d.emplace_back(v_x3Dc[i](0), v_x3Dc[i](1), v_x3Dc[i](2));
        }
    }
    frame->vio_p_wb = back_frame->p_wb;
    frame->vio_q_wb = back_frame->q_wb;
    frame->p_wb = back_frame->p_wb;
    frame->q_wb = back_frame->q_wb;
    return frame;
}
