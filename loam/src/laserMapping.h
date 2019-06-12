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
#include "lidarFactor.hpp"
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

struct LidarState{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double lidar_time = -1;
    double BackTime_i = -1;
    double BackTime_j = -1;
    Eigen::Quaterniond wvio_Qbi;
    Eigen::Vector3d wvio_tbi;
    Eigen::Vector3d wvio_veci;
    Eigen::Vector3d bias_acc_i;
    Eigen::Vector3d bias_gyr_i;
    Eigen::Quaterniond wvio_Qbj;
    Eigen::Vector3d wvio_tbj;
    Eigen::Vector3d wvio_vecj;
    Eigen::Vector3d bias_acc_j;
    Eigen::Vector3d bias_gyr_j;
    Eigen::VecVector3d v_gyr, v_acc;
    bool exist_imu = false;
    IntegrationBasePtr imupreinte;
};
SMART_PTR(LidarState)
