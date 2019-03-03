#pragma once
#include "util.h"

class ConfigLoader {
public:
    struct Param {
        int num_camera;
        std::vector<int> width;
        std::vector<int> height;
        std::vector<std::string> cam_model;
        std::vector<std::vector<double>> intrinsic_master;
        std::vector<std::vector<double>> distortion_master;
        std::vector<Eigen::Vector3d> p_bc;
        std::vector<Sophus::SO3d> q_bc;
        std::vector<bool> b_slave;
        std::vector<std::vector<double>> intrinsic_slave;
        std::vector<std::vector<double>> distortion_slave;
        std::vector<Eigen::Vector3d> p_rl;
        std::vector<Sophus::SO3d> q_rl;

        double acc_noise;
        double gyr_noise;
        double acc_bias_noise;
        double gyr_bias_noise;
        double gravity_magnitude;

        double clahe_parameter;
        int fast_threshold;
        int min_dist;
        double frontend_pub_period;
        float Fundamental_reproj_threshold;

        double max_solver_time_in_seconds;
        int max_num_iterations;
        double keyframe_parallax;
        double cv_huber_loss_parameter;
        int sliding_window_size;
        double triangulate_default_depth;
        double max_imu_sum_t;
        int min_init_stereo_num;

        int enable_reloc;
        int enable_pose_faster;
        int estimate_extrinsic;

        std::string voc_filename;
        std::string brief_pattern_file;
    };
    static Param Load(const std::string& config_file);
private:
    static void Log(const Param& param);
};
