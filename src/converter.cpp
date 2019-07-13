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
        Pl = cam_master->BackProject(Eigen::Vector2d(feat_frame->pt[i].x, feat_frame->pt[i].y));
        backend_frame->pt_normal_plane.emplace_back(Pl);
        if(stereo_frame->pt_r[i].x == -1) {
            backend_frame->pt_r_normal_plane.emplace_back(-1, -1, 0);
        }
        else {
            Pr = cam_slave->BackProject(Eigen::Vector2d(stereo_frame->pt_r[i].x, stereo_frame->pt_r[i].y));
            backend_frame->pt_r_normal_plane.emplace_back(Pr);
        }
    }

    backend_frame->v_gyr = v_gyr;
    backend_frame->v_acc = v_acc;
    backend_frame->v_imu_timestamp = v_imu_timestamp;
    return backend_frame;
}

BackEnd::FramePtr Converter::Convert(FeatureTracker::FramePtr feat_frame, CameraPtr camera,
                                     const cv::Mat& depth_iamge, double depth_units,
                                     const Sophus::SO3d& q_rl, const Eigen::Vector3d& p_rl,
                                     const Eigen::VecVector3d& v_gyr, const Eigen::VecVector3d& v_acc,
                                     const std::vector<double>& v_imu_timestamp)
{
     BackEnd::FramePtr backend_frame(new BackEnd::Frame);
     backend_frame->timestamp = feat_frame->timestamp;
     backend_frame->pt_id = feat_frame->pt_id;

     for(int i = 0, n = feat_frame->pt.size(); i < n; ++i) {
         Eigen::Vector3d Pl;
         Pl = camera->BackProject(Eigen::Vector2d(feat_frame->pt[i].x, feat_frame->pt[i].y));
         backend_frame->pt_normal_plane.emplace_back(Pl);
         double depth = depth_iamge.at<uint16_t>(feat_frame->pt[i]) * depth_units;

         if(depth) {
             Eigen::Vector3d x3Dl = Pl * depth;
             Eigen::Vector3d x3Dr = q_rl * x3Dl + p_rl;
             x3Dr /= x3Dr(2);
             backend_frame->pt_r_normal_plane.emplace_back(x3Dr);
         }
         else {
             backend_frame->pt_r_normal_plane.emplace_back(-1, -1, 0);
         }
     }

     backend_frame->v_gyr = v_gyr;
     backend_frame->v_acc = v_acc;
     backend_frame->v_imu_timestamp = v_imu_timestamp;
     return backend_frame;
}

PLBackEnd::FramePtr Converter::Convert(FeatureTracker::FramePtr f_frame, StereoMatcher::FramePtr s_frame,
                                       LineTracker::FramePtr l_frame, LineStereoMatcher::FramePtr ls_frame,
                                       CameraPtr cam_m, CameraPtr cam_s, const Eigen::VecVector3d& v_gyr,
                                       const Eigen::VecVector3d& v_acc, const std::vector<double>& v_imu_t)
{
    PLBackEnd::FramePtr frame(new PLBackEnd::Frame);
    frame->timestamp = f_frame->timestamp;
    frame->pt_id = f_frame->pt_id;
    for(int i = 0, n = f_frame->pt_id.size(); i < n; ++i) {
        Eigen::Vector3d Pl, Pr;
        Pl = cam_m->BackProject(Eigen::Vector2d(f_frame->pt[i].x, f_frame->pt[i].y));
        frame->pt_normal_plane.emplace_back(Pl);
        if(s_frame->pt_r[i].x == -1) {
            frame->pt_r_normal_plane.emplace_back(-1, -1, 0);
        }
        else {
            Pr = cam_s->BackProject(Eigen::Vector2d(s_frame->pt_r[i].x, s_frame->pt_r[i].y));
            frame->pt_r_normal_plane.emplace_back(Pr);
        }
    }

    frame->line_id = l_frame->v_line_id;
    for(int i = 0, n = l_frame->v_line_id.size(); i < n; ++i) {
        Eigen::Vector3d Pl, Pr, Ql, Qr;
        Pl = cam_m->BackProject(Eigen::Vector2d(l_frame->v_lines[i].startPointX, l_frame->v_lines[i].startPointY));
        Ql = cam_m->BackProject(Eigen::Vector2d(l_frame->v_lines[i].endPointX, l_frame->v_lines[i].endPointY));
        frame->line_spt_n.emplace_back(Pl);
        frame->line_ept_n.emplace_back(Ql);
        if(ls_frame->v_lines_r[i].startPointX == -1) {
            frame->line_spt_r_n.emplace_back(-1, -1, 0);
            frame->line_ept_r_n.emplace_back(-1, -1, 0);
        }
        else {
            Pr = cam_s->BackProject(Eigen::Vector2d(ls_frame->v_lines_r[i].startPointX, ls_frame->v_lines_r[i].startPointY));
            Qr = cam_s->BackProject(Eigen::Vector2d(ls_frame->v_lines_r[i].endPointX, ls_frame->v_lines_r[i].endPointY));
            frame->line_spt_r_n.emplace_back(Pr);
            frame->line_ept_r_n.emplace_back(Qr);
        }
    }

    frame->v_gyr = v_gyr;
    frame->v_acc = v_acc;
    frame->v_imu_timestamp = v_imu_t;
    return frame;
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
