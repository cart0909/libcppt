#pragma once
#include <math.h>
#include <vector>
#include <vloam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include "lidarFactor.h"
#include "vloam_velodyne/common.h"
#include "add_msg/RelativePoseIMU.h"
#include "add_msg/Intervalimu.h"
#include "vins/imu_factor.h"
#include "vins/integration_base.h"
#include "util.h"
double acc_n = -1;
double gyr_n = -1;
double acc_w = -1;
double gyr_w = -1;
Eigen::Vector3d gw(0,0,0);
double g_norm = -1;
bool systemInited = false;
int sequence = 0;
double parameters_j[7] = {0, 0, 0, 1, 0, 0, 0}; //parameters j which represent w_T_bodyj(curr)
double parameters_i[7] = {0, 0, 0, 1, 0, 0, 0}; //parameters i which represent w_T_bodyi
double parameters_speed_bias_i[9];
double parameters_speed_bias_j[9];
//Wmap_T_bodycurr
Eigen::Map<Sophus::SE3d> w_T_bodyj(parameters_j);
Eigen::Quaterniond q_w_lidarj;
Eigen::Vector3d t_w_lidarj;

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);
Sophus::SE3d lidar_T_body;
Sophus::SE3d w_T_lidarj_last;

struct LidarState{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double lidar_time = -1;
    double BackTime_i = -1;
    double BackTime_j = -1;

    Sophus::SO3d wmap_Qbi;
    Eigen::Vector3d wmap_tbi;
    Eigen::Vector3d wmap_veci;
    Eigen::Vector3d bias_acc_i;
    Eigen::Vector3d bias_gyr_i;
    Sophus::SO3d wmap_Qbj;
    Eigen::Vector3d wmap_tbj;
    Eigen::Vector3d wmap_vecj;
    Eigen::Vector3d bias_acc_j;
    Eigen::Vector3d bias_gyr_j;
    Eigen::VecVector3d v_gyr, v_acc;
    bool exist_imu = false;
    IntegrationBasePtr imupreinte;
};
SMART_PTR(LidarState)
