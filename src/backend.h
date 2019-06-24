#pragma once

#include <thread>
#include <atomic>
#include <condition_variable>
#include "util.h"
#include "vins/integration_base.h"
#include "vins/marginalization_factor.h"
#include "add_msg/ImuPredict.h"
#include "sensor_msgs/PointCloud2.h"
#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
class BackEnd {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum State {
        NEED_INIT,
        CV_ONLY,
        TIGHTLY
    };

    enum MarginType {
        MARGIN_SECOND_NEW,
        MARGIN_OLD
    };

    struct Frame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        virtual ~Frame() {}
        uint64_t id;
        double timestamp;
        std::vector<uint64_t>  pt_id;
        Eigen::VecVector3d pt_normal_plane, pt_r_normal_plane;
        Sophus::SO3d    q_wb;
        Eigen::Vector3d p_wb;
        Eigen::Vector3d v_wb;
        Eigen::Vector3d ba;
        Eigen::Vector3d bg;
        Eigen::VecVector3d v_gyr, v_acc;
        std::vector<double> v_imu_timestamp;
        IntegrationBasePtr imupreinte;
        double td; // time delay

        //lidarInfo
        double lidar_time = -1;
        int lidarInfo = -1; //1:lidartime <=imgae; 2:lidartime > image.
        Eigen::Vector3d p_wb_lidar;
        Sophus::SO3d    q_wb_lidar;
        pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;
        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast;
        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast;
        std::vector<Eigen::Vector3d> curr_point_edge;
        std::vector<Eigen::Vector3d> last_point_a_edge;
        std::vector<Eigen::Vector3d> last_point_b_edge;

        std::vector<Eigen::Vector3d> curr_point_plane;
        std::vector<Eigen::Vector3d> last_point_a_plane;
        std::vector<Eigen::Vector3d> last_point_b_plane;
        std::vector<Eigen::Vector3d> last_point_c_plane;
    };
    SMART_PTR(Frame)

    struct Feature {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                Feature(uint64_t feat_id_, int start_id_)
            : feat_id(feat_id_), start_id(start_id_),
              inv_depth(-1.0f), last_time(-1.0f), last_r_time(-1.0f) {}

        uint64_t feat_id;
        int start_id;

        inline int CountNumMeas(int sw_idx) const {
            int num_meas = 0;
            for(int i = 0, n = pt_n_per_frame.size(); i < n; ++i) {
                if(start_id + i > sw_idx)
                    break;
                if(pt_r_n_per_frame[i](2) != 0)
                    num_meas += 2;
                else
                    num_meas += 1;
            }
            return num_meas;
        }

        Eigen::DeqVector3d pt_n_per_frame;
        Eigen::DeqVector3d pt_r_n_per_frame;
        double inv_depth;

        // estimate time delay
        Eigen::DeqVector3d velocity_per_frame;
        Eigen::DeqVector3d velocity_r_per_frame;
        Eigen::Vector3d last_pt_n;
        Eigen::Vector3d last_pt_r_n;
        double last_time, last_r_time;
    };
    SMART_PTR(Feature)

    BackEnd();
    BackEnd(double focal_length_,
            double gyr_n_, double acc_n_,
            double gyr_w_, double acc_w_,
            const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
            const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_,
            double gravity_magnitude_, int window_size_, double min_parallax_,
            double max_solver_time_in_seconds_, int max_num_iterations_,
            double cv_huber_loss_parameter_, double triangulate_default_depth_,
            double max_imu_sum_t_, int min_init_stereo_num_, int estimate_extrinsic,
            int estimate_td, double init_td);
    virtual ~BackEnd();

    inline void SubVIOTwb(std::function<void(double, const Sophus::SE3d)> callback) {
        pub_vio_Twb.emplace_back(callback);
    }

    inline void SubVIOTwc(std::function<void(double, const Sophus::SE3d)> callback) {
        pub_vio_Twc.emplace_back(callback);
    }

    inline void SubKeyFrame(std::function<void(FramePtr, const Eigen::VecVector3d&)> callback) {
        pub_keyframe.emplace_back(callback);
    }

    inline void SubFrame(std::function<void(FramePtr)> callback) {
        pub_frame.emplace_back(callback);
    }

    inline void SubMapPoints(std::function<void(const Eigen::VecVector3d&)> callback) {
        pub_mappoints.emplace_back(callback);
    }

    inline void SubIMUPreintInfo(std::function<void(double, const add_msg::ImuPredict&)> callback) {
        ImuPreintInfo = callback;
    }

    inline void ResetRequest() {
        request_reset_flag = true;
    }

    void ProcessFrame(FramePtr frame);
    void ProcessFrame(FramePtr frame,
                      std::pair<sensor_msgs::PointCloud2ConstPtr, sensor_msgs::PointCloud2ConstPtr> lidarInfo,
                      bool &start_this_frame, Eigen::VecVector3d& lidar_acc, Eigen::VecVector3d& lidar_gyro, std::vector<double>& lidar_imut);
    bool GetNewKeyFrameAndMapPoints(FramePtr& keyframe, Eigen::VecVector3d& v_x3Dc);
    State state;

    inline double GetTd() const {
        return time_delay;
    }

protected:
    MarginType AddFeaturesCheckParallax(FramePtr frame);
    virtual void AddFeatures(FramePtr frame, int& last_track_num);
    void SlidingWindow();
    virtual void SlidingWindowOld();
    virtual void SlidingWindowSecondNew();
    virtual int Triangulate(int sw_idx);
    virtual void Reset();
    virtual void SolveBA();
    virtual void SolveBAImu();
    void SolvePnP(FramePtr frame);
    void TransLidarPointToImage(FramePtr frame, std::pair<sensor_msgs::PointCloud2ConstPtr, sensor_msgs::PointCloud2ConstPtr> lidarInfo);
    void FindEdgeNear3DPoint(FramePtr last_frame, FramePtr frame);
    void FindPlaneNear3DPoint(FramePtr last_frame, FramePtr frame);
    void TransformToStart(PointType const *const pi, PointType *const po, Sophus::SE3d &lidari_T_lidarj);
    bool GyroBiasEstimation();
    Sophus::SO3d InitFirstIMUPose(const Eigen::VecVector3d& v_acc);
    void PredictNextLidarPose(FramePtr cur_frame, Eigen::VecVector3d& lidar_acc, Eigen::VecVector3d& lidar_gyro, std::vector<double>& lidar_imut);
    virtual void PredictNextFramePose(FramePtr ref_frame, FramePtr cur_frame);
    virtual void Marginalize();
    virtual void Publish();

    double focal_length;
    Eigen::Vector3d p_rl, p_bc;
    Sophus::SO3d    q_rl, q_bc;
    Sophus::SE3d Tlb;
    double SCAN_PERIOD = 0.1;
    const double DISTANCE_SQ_THRESHOLD = 25;
    const double NEARBY_SCAN = 2.5;
    double gyr_n, acc_n, gyr_w, acc_w;
    Eigen::Matrix3d gyr_noise_cov;
    Eigen::Matrix3d acc_noise_cov;
    Eigen::Matrix6d gyr_acc_noise_cov;
    Eigen::Matrix3d gyr_bias_cov;
    Eigen::Matrix3d acc_bias_cov;
    Eigen::Matrix6d acc_gyr_bias_invcov;

    std::map<uint64_t, Feature> m_features;
    int window_size;
    std::deque<FramePtr> d_frames; // [ 0,  1, ..., 8 ,         9 |  10] size 11
    //  kf  kf      kf  second new   new
    uint64_t next_frame_id;

    double min_parallax;
    MarginType marginalization_flag;

    // ceres data
    virtual void data2double();
    virtual void double2data();
    double* para_pose; // Twb
    double* para_speed_bias; // vwb bg ba
    size_t  para_features_capacity = 1000;
    double* para_features; // inv_z
    int enable_estimate_extrinsic;
    double para_ex_bc[7];
    double para_ex_sm[7];

    // maintain system
    std::atomic<bool> request_reset_flag;

    double last_imu_t;
    double gravity_magnitude;
    Eigen::Vector3d gw;

    std::vector<double*> para_margin_block;
    MarginalizationInfo* last_margin_info;

    Eigen::VecVector3d margin_mps;
    std::vector<std::function<void(double, const Sophus::SE3d)>> pub_vio_Twb;
    std::vector<std::function<void(double, const Sophus::SE3d)>> pub_vio_Twc;
    std::vector<std::function<void(FramePtr, const Eigen::VecVector3d&)>> pub_keyframe;
    std::vector<std::function<void(const Eigen::VecVector3d&)>> pub_mappoints;
    std::vector<std::function<void(FramePtr)>> pub_frame;
    std::function<void(double, const add_msg::ImuPredict&)> ImuPreintInfo;
    // optimize parameters
    double max_solver_time_in_seconds;
    int max_num_iterations;
    double cv_huber_loss_parameter;
    double triangulate_default_depth;
    double max_imu_sum_t;
    int min_init_stereo_num;

    bool estimate_time_delay;
    double time_delay;
    double para_Td[1];

    FramePtr new_keyframe;
    add_msg::ImuPredict imuinfo_tmp;
    Eigen::VecVector3d v_new_keyframe_x3Dc;
};
SMART_PTR(BackEnd)
