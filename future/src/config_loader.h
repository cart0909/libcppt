#pragma once
#include "util.h"

class ConfigLoader {
public:
    struct Param {
        int num_camera;
        std::vector<int> width;
        std::vector<int> height;
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
    };
    static Param Load(const std::string& config_file);
private:
    static void Log(const Param& param);
};
