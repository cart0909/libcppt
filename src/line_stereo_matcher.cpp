#include "line_stereo_matcher.h"

LineStereoMatcher::LineStereoMatcher(CameraPtr cam_l_, CameraPtr cam_r_): cam_l(cam_l_), cam_r(cam_r_) {
    lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    fld = cv::ximgproc::createFastLineDetector(32, 1.414, 50, 30, 3, true);
}

void LineStereoMatcher::Process(LineTracker::FramePtr l_frame, FramePtr r_frame) {
    if(l_frame->v_lines.empty()) {
        r_frame->v_lines_r.clear();
        r_frame->desc_r.release();
        return;
    }

     std::vector<cv::line_descriptor::KeyLine> v_klines_r = r_frame->v_lines_r;
     r_frame->v_lines_r.clear();

     if(v_klines_r.empty()) {
         for(int i = 0, n = l_frame->v_line_id.size(); i < n; ++i) {
             cv::line_descriptor::KeyLine kl;
             kl.startPointX = -1;
             kl.startPointY = -1;
             kl.endPointX = -1;
             kl.endPointY = -1;
             r_frame->v_lines_r.emplace_back(kl);
         }
         return;
     }

     std::vector<int> matches_lr;
     LRMatch(l_frame->desc, r_frame->desc_r, 0.75, matches_lr);

     for(int i = 0, n = matches_lr.size(); i < n; ++i) {
         if(matches_lr[i] == -1) {
             cv::line_descriptor::KeyLine kl;
             kl.startPointX = -1;
             kl.startPointY = -1;
             kl.endPointX = -1;
             kl.endPointY = -1;
             r_frame->v_lines_r.emplace_back(kl);
         }
         else {
             cv::line_descriptor::KeyLine kl_tmp = l_frame->v_lines[i];
             cv::line_descriptor::KeyLine kr_tmp = v_klines_r[matches_lr[i]];
             //TODO:: save undistort point for coverter used.
             Eigen::Vector3d undis_kl_ep;
             Eigen::Vector3d undis_kl_sp;
             Eigen::Vector3d undis_kr_ep;
             Eigen::Vector3d undis_kr_sp;
             undis_kl_sp = cam_l->BackProject(Eigen::Vector2d(kl_tmp.startPointX, kl_tmp.startPointY));
             undis_kl_ep = cam_l->BackProject(Eigen::Vector2d(kl_tmp.endPointX, kl_tmp.endPointY));
             undis_kr_sp = cam_r->BackProject(Eigen::Vector2d(kr_tmp.startPointX, kr_tmp.startPointY));
             undis_kr_ep = cam_r->BackProject(Eigen::Vector2d(kr_tmp.endPointX, kr_tmp.endPointY));
             Eigen::Vector2d spl;
             Eigen::Vector2d epl;
             Eigen::Vector2d spr;
             Eigen::Vector2d epr;
             cam_l->Project(undis_kl_sp, spl);
             cam_l->Project(undis_kl_ep, epl);
             cam_r->Project(undis_kr_sp, spr);
             cam_r->Project(undis_kr_ep, epr);
             double overlap = util::f2fLineSegmentOverlap(spl.head<2>(), epl.head<2>(), spr.head<2>(), epr.head<2>());

             //diff of angle
             Eigen::Vector2d Ii = (spl - epl) / (spl - epl).squaredNorm();
             Eigen::Vector2d Ij = (spr - epr) / (spr - epr).squaredNorm();
             double ij_cross = (Eigen::Vector3d(Ii.x(), Ii.y(), 0).cross(Eigen::Vector3d(Ij.x(), Ij.y(), 0))).norm();
             double ij_dot = Ij.dot(Ii);
             double theta = std::atan2(ij_cross, ij_dot) * 180 / M_PI;

             //find the match inlier keyline and insert value to frame.
             if(overlap > 0.65 && theta < 5){
                 r_frame->v_lines_r.emplace_back(v_klines_r[matches_lr[i]]);
             }
             else{
                 cv::line_descriptor::KeyLine kl;
                 kl.startPointX = -1;
                 kl.startPointY = -1;
                 kl.endPointX = -1;
                 kl.endPointY = -1;
                 r_frame->v_lines_r.emplace_back(kl);
                 matches_lr[i] = -1;
             }
         }
     }

#if 0
     std::vector<cv::DMatch> dmatches;

     for(int i = 0, n = matches_lr.size(); i < n; ++i) {
         if(matches_lr[i] == -1) {
             continue;
         }
         cv::DMatch dmatch;
         dmatch.queryIdx = i;
         dmatch.trainIdx = matches_lr[i];
         dmatch.distance = 0;
         dmatch.imgIdx = 0;
         dmatches.emplace_back(dmatch);
     }

     cv::Mat result;
     cv::Mat show_img_l, show_img_r;
     cv::cvtColor(l_frame->img, show_img_l, CV_GRAY2BGR);
     cv::cvtColor(r_frame->img_r, show_img_r, CV_GRAY2BGR);
     std::vector<char> mask(dmatches.size(), 1);
     cv::line_descriptor::drawLineMatches(show_img_l, l_frame->v_lines, show_img_r, v_klines_r,
                                          dmatches, result,
                                          cv::Scalar::all(-1), cv::Scalar::all(-1), mask,
                                          cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT );
     cv::imshow("result", result);
     cv::waitKey(1);
#endif

    return;
}

LineStereoMatcher::FramePtr LineStereoMatcher::InitFrame(const cv::Mat& img_r) {
    FramePtr r_frame(new Frame);
    r_frame->img_r = img_r;
    r_frame->v_lines_r = DetectLineFeatures(r_frame->img_r);

    if(r_frame->v_lines_r.empty()) {
        return r_frame;
    }

    Tracer::TraceBegin("lbd_r");
    lbd->compute(r_frame->img_r, r_frame->v_lines_r, r_frame->desc_r);
    Tracer::TraceEnd();
    return r_frame;
}

std::vector<cv::line_descriptor::KeyLine> LineStereoMatcher::DetectLineFeatures(const cv::Mat& img) {
    std::vector<cv::Vec4f> v_lines;
    std::vector<cv::line_descriptor::KeyLine> v_klines;
    Tracer::TraceBegin("fld_r");
    fld->detect(img, v_lines);
    Tracer::TraceEnd();

    if(v_lines.size() > 50) {
        std::sort(v_lines.begin(), v_lines.end(), [](const cv::Vec4f& lhs, const cv::Vec4f& rhs) {
            double ldx = lhs[0] - lhs[2];
            double ldy = lhs[1] - lhs[3];
            double rdx = rhs[0] - rhs[2];
            double rdy = rhs[1] - rhs[3];
            return (ldx * ldx + ldy * ldy) > (rdx * rdx + rdy * rdy);
        });
        v_lines.resize(50);
    }

    for(int i = 0, n = v_lines.size(); i < n; ++i) {
        auto& line = v_lines[i];
        cv::line_descriptor::KeyLine kl;
        double octave_scale = 1.0f;
        int octave_idx = 0;
        kl.startPointX = line[0] * octave_scale;
        kl.startPointY = line[1] * octave_scale;
        kl.endPointX = line[2] * octave_scale;
        kl.endPointY = line[3] * octave_scale;

        kl.sPointInOctaveX = line[0];
        kl.sPointInOctaveY = line[1];
        kl.ePointInOctaveX = line[2];
        kl.ePointInOctaveY = line[3];

        double dx = line[2] - line[0];
        double dy = line[3] - line[1];
        kl.lineLength = std::sqrt(dx * dx + dy * dy);
        kl.angle = std::atan2(dy, dx);
        kl.class_id = i;
        kl.octave = octave_idx;
        kl.size = dx * dy;
        kl.pt = cv::Point2f((line[0] + line[2]) * 0.5, (line[1] + line[3]) * 0.5);

        kl.response = kl.lineLength / std::max(img.rows, img.cols);
        cv::LineIterator li(img, cv::Point2f(line[0], line[1]), cv::Point2f(line[2], line[3]));
        kl.numOfPixels = li.count;

        v_klines.emplace_back(kl);
    }

    return v_klines;
}

int LineStereoMatcher::Match(const cv::Mat& desc1, const cv::Mat& desc2, float nnr, std::vector<int>& matches12) {
    int num_matches = 0;
    matches12.resize(desc1.rows, -1);

    std::vector<std::vector<cv::DMatch>> dmatches;
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    matcher->knnMatch(desc1, desc2, dmatches, 2);

    for(int i = 0, n = matches12.size(); i < n; ++i) {
        if(dmatches[i][0].distance < dmatches[i][1].distance * nnr) {
            matches12[i] = dmatches[i][0].trainIdx;
            ++num_matches;
        }
    }

    return num_matches;
}

int LineStereoMatcher::LRMatch(const cv::Mat& desc1, const cv::Mat& desc2, float nnr, std::vector<int>& matches12) {
    std::vector<int> matches21;
    int num_matches = Match(desc1, desc2, nnr, matches12);
    Match(desc2, desc1, nnr, matches21);

    for(int i1 = 0; i1 < matches12.size(); ++i1) {
        int i2 = matches12[i1];
        if(i2 >= 0 && matches21[i2] != i1) {
            matches12[i1] = -1;
            --num_matches;
        }
    }
    return num_matches;
}
