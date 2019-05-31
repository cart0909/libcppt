#include "rgbd_system.h"
#include "rgbd_backend.h"
#include "converter.h"

RGBDSystem::RGBDSystem(const std::string& config_file) {
    cv::setNumThreads(0);
    reset_flag = false;
    param = ConfigLoader::Load(config_file);
    InitCameraParameters();

    // create virtual right caemra
    param.q_rl[0].setQuaternion(Eigen::Quaterniond(1, 0, 0, 0));
    param.p_rl[0] << -0.1, 0, 0;

    feature_tracker = std::make_shared<FeatureTracker>(cam_m, param.clahe_parameter, param.fast_threshold,
                                                       param.min_dist, param.Fundamental_reproj_threshold);

    backend = std::make_shared<RGBDBackEnd>(cam_m->f(), param.gyr_noise, param.acc_noise,
                                            param.gyr_bias_noise, param.acc_bias_noise,
                                            param.p_rl[0], param.p_bc[0], param.q_rl[0], param.q_bc[0],
                                            param.gravity_magnitude, param.sliding_window_size, param.keyframe_parallax,
                                            param.max_solver_time_in_seconds, param.max_num_iterations, param.cv_huber_loss_parameter,
                                            param.triangulate_default_depth, param.max_imu_sum_t, param.min_init_stereo_num, param.estimate_extrinsic,
                                            param.estimate_td, param.init_td);

    if(param.enable_reloc)
        reloc = std::make_shared<Relocalization>(param.voc_filename, param.brief_pattern_file, cam_m,
                                                 param.q_bc[0], param.p_bc[0]);
    if(reloc) {
        backend->SubVIOTwc(std::bind(&Relocalization::UpdateVIOPose, reloc,
                                     std::placeholders::_1,
                                     std::placeholders::_2));
        backend->SubVIOTwb(std::bind(&Relocalization::UpdateVIOPoseWB, reloc,
                                     std::placeholders::_1,
                                     std::placeholders::_2));
        reloc_thread = std::thread(&RGBDSystem::RelocProcess, this);
    }

    frontend_thread = std::thread(&RGBDSystem::FrontEndProcess, this);
    backend_thread = std::thread(&RGBDSystem::BackEndProcess, this);
}

RGBDSystem::~RGBDSystem() {}

void RGBDSystem::InitCameraParameters() {
    if(param.cam_model[0] == "PINHOLE") {
        cam_m = CameraPtr(new Pinhole(param.width[0], param.height[0],
                                      param.intrinsic_master[0][0], param.intrinsic_master[0][1],
                                      param.intrinsic_master[0][2], param.intrinsic_master[0][3],
                                      param.distortion_master[0][0], param.distortion_master[0][1],
                                      param.distortion_master[0][2], param.distortion_master[0][3]));
    }
    else if(param.cam_model[0] == "FISHEYE") {
        cam_m = CameraPtr(new Fisheye(param.width[0], param.height[0],
                                      param.intrinsic_master[0][0], param.intrinsic_master[0][1],
                                      param.intrinsic_master[0][2], param.intrinsic_master[0][3],
                                      param.distortion_master[0][0], param.distortion_master[0][1],
                                      param.distortion_master[0][2], param.distortion_master[0][3]));
    }
    else if(param.cam_model[0] == "OMNI") {
        cam_m = CameraPtr(new Omni(param.width[0], param.height[0],
                                   param.intrinsic_master[0][0], param.intrinsic_master[0][1],
                                   param.intrinsic_master[0][2], param.intrinsic_master[0][3],
                                   param.intrinsic_master[0][4],
                                   param.distortion_master[0][0], param.distortion_master[0][1],
                                   param.distortion_master[0][2], param.distortion_master[0][3]));
    }
}

void RGBDSystem::BackEndProcess() {
    while(1) {
        Eigen::VecVector3d v_gyr, v_acc;
        std::vector<double> v_imu_t;
        FeatureTracker::FramePtr feat_frame;
        cv::Mat depth_img;
        double img_t = -1.0f;
        double td = backend->GetTd();
        std::unique_lock<std::mutex> lock(mtx_backend);
        cv_backend.wait(lock, [&] {
            while(1) {
                if(backend_buffer_img.empty() || backend_buffer_imu_t.empty())
                    return false;

                if(backend_buffer_imu_t.back() < backend_buffer_img.front().first->timestamp + td)
                    return false;

                if(backend_buffer_imu_t.front() > backend_buffer_img.front().first->timestamp + td) {
                    backend_buffer_img.pop_front();
                    continue;
                }

                feat_frame = backend_buffer_img.front().first;
                depth_img = backend_buffer_img.front().second;
                img_t = feat_frame->timestamp;
                backend_buffer_img.pop_front();

                while(backend_buffer_imu_t.front() < img_t + td) {
                    v_gyr.emplace_back(backend_buffer_gyr.front());
                    v_acc.emplace_back(backend_buffer_acc.front());
                    v_imu_t.emplace_back(backend_buffer_imu_t.front());
                    backend_buffer_gyr.pop_front();
                    backend_buffer_acc.pop_front();
                    backend_buffer_imu_t.pop_front();
                }

                return true;
            }
        });
        lock.unlock();
        backend_busy = true;

        feature_tracker->ExtractFAST(feat_frame);
        PubTrackingImg(feat_frame);
        BackEnd::FramePtr back_frame = Converter::Convert(feat_frame, cam_m, depth_img, param.depth_units,
                                                          param.q_rl[0], param.p_rl[0], v_gyr, v_acc, v_imu_t);

        back_frame->td = td;
        backend->ProcessFrame(back_frame);

        if(reloc) {
            BackEnd::FramePtr new_keyframe;
            Eigen::VecVector3d v_x3Dc;
            if(backend->GetNewKeyFrameAndMapPoints(new_keyframe, v_x3Dc)) {
                mtx_reloc.lock();
                reloc_buffer_keyframe.emplace_back(new_keyframe, v_x3Dc);
                mtx_reloc.unlock();
                cv_reloc.notify_one();
            }

            mtx_reloc.lock();
            reloc_buffer_frame.emplace_back(feat_frame, back_frame);
            mtx_reloc.unlock();
            cv_reloc.notify_one();
        }
        backend_busy = false;
    }
}
