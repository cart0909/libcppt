// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>
#include "vloam_velodyne/common.h"
#include "lidarFactor.hpp"
#include "sophus/se3.hpp"
#include "ceres/local_parameterization_se3.h"
#include "util.h"
#include "add_msg/ImuPredict.h"
#include "add_msg/RelativePoseIMU.h"
#include "add_msg/Intervalimu.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

//#include "local_parameterization.h"
#define DISTORTION 0
int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;
Eigen::Vector3d gw(-1, -1, -1);
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to camera_init frame
Sophus::SO3d q_w_curr;
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_last_curr(x, y, z, w), t_last_curr
Eigen::Vector3d para_P; //tlast_cur
Sophus::SO3d para_R;    //Rlast_cur
double* para_pose; //Tlast_cur
add_msg::ImuPredict wvioTbi;
add_msg::ImuPredict wvioTbj;

Sophus::SE3d lidar_T_body;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::queue<add_msg::ImuPredictConstPtr> fullPredictPoseBuf;
std::queue<sensor_msgs::ImuConstPtr> fullImuRawBuf;
std::mutex mBuf;
std::mutex mBuf_imu_raw;
std::mutex mBuf_imu_preint;

// undistort lidar point
//TODO::Using visual IMU predict this pose.
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, para_R.unit_quaternion());
    Eigen::Vector3d t_point_last = s * para_P;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame
//TODO::Using visual IMU predict this pose.
void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = para_R.inverse().unit_quaternion() * (un_point - para_P);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}


void Sync_callback(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2,
                   const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2,
                   const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2){
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    surfLessFlatBuf.push(surfPointsLessFlat2);
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void VisualImuInfoHandler(const add_msg::ImuPredictConstPtr &imu_info)
{
    mBuf_imu_preint.lock();
    fullPredictPoseBuf.push(imu_info);
    mBuf_imu_preint.unlock();
}

void ImuRawInfoHandler(const sensor_msgs::ImuConstPtr& imu_msg)
{
    mBuf_imu_raw.lock();
    fullImuRawBuf.push(imu_msg);
    mBuf_imu_raw.unlock();
}

void ImuIntegrationForNextPose(add_msg::ImuPredict& vpose_match, add_msg::RelativePoseIMU& poseij_imu,
                               std::vector<sensor_msgs::Imu>& imu_msg_predict)
{
    Sophus::SO3d qwb;
    Eigen::Quaterniond tmp_qwb;
    double t0 = vpose_match.BackTime;
    //insert Q
    tmp_qwb.x() = vpose_match.pose.orientation.x;
    tmp_qwb.y() = vpose_match.pose.orientation.y;
    tmp_qwb.z() = vpose_match.pose.orientation.z;
    tmp_qwb.w() = vpose_match.pose.orientation.w;
    qwb.setQuaternion(tmp_qwb);
    //insert P
    Eigen::Vector3d pwb(vpose_match.pose.position.x, vpose_match.pose.position.y, vpose_match.pose.position.z);

    //insert V
    Eigen::Vector3d vwb(vpose_match.velocity.x, vpose_match.velocity.y, vpose_match.velocity.z);

    //insert bg/ba
    Eigen::Vector3d bg(vpose_match.bias_gyro.x, vpose_match.bias_gyro.y, vpose_match.bias_gyro.z);
    Eigen::Vector3d ba(vpose_match.bias_acc.x, vpose_match.bias_acc.y, vpose_match.bias_acc.z);
    //insert last raw imu info
    Eigen::Vector3d gyr_0(vpose_match.gyr_0.x, vpose_match.gyr_0.y, vpose_match.gyr_0.z);
    Eigen::Vector3d acc_0(vpose_match.acc_0.x, vpose_match.acc_0.y, vpose_match.acc_0.z);
    //Eigen::Vector3d gw(0, 0, 9.81007);


    //finsish predict which need to update the p, v, r, bg, ba, header_timestamp, backtime, last_imuraw_info
    for(int i = 0, n = imu_msg_predict.size(); i < n; ++i) {
        double t = imu_msg_predict[i].header.stamp.toSec(), dt = t - t0;
        Eigen::Vector3d gyr = Eigen::Vector3d(imu_msg_predict[i].angular_velocity.x, imu_msg_predict[i].angular_velocity.y, imu_msg_predict[i].angular_velocity.z);
        Eigen::Vector3d acc = Eigen::Vector3d(imu_msg_predict[i].linear_acceleration.x, imu_msg_predict[i].linear_acceleration.y, imu_msg_predict[i].linear_acceleration.z);
        Eigen::Vector3d un_acc_0 = qwb * (acc_0 - ba) - gw;
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr) - bg;
        qwb = qwb * Sophus::SO3d::exp(un_gyr * dt);
        Eigen::Vector3d un_acc_1 = qwb * (acc - ba) - gw;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        pwb += dt * vwb + 0.5 * dt * dt * un_acc;
        vwb += dt * un_acc;
        gyr_0 = gyr;
        acc_0 = acc;
        t0 = t;
    }
    wvioTbj.header = vpose_match.header;
    wvioTbj.pose.position.x = pwb.x();
    wvioTbj.pose.position.y = pwb.y();
    wvioTbj.pose.position.z = pwb.z();

    wvioTbj.bias_acc = vpose_match.bias_acc;
    wvioTbj.bias_gyro = vpose_match.bias_gyro;

    wvioTbj.pose.orientation.x = qwb.unit_quaternion().x();
    wvioTbj.pose.orientation.y = qwb.unit_quaternion().y();
    wvioTbj.pose.orientation.z = qwb.unit_quaternion().z();
    wvioTbj.pose.orientation.w = qwb.unit_quaternion().w();

    wvioTbj.velocity.x = vwb.x();
    wvioTbj.velocity.y = vwb.y();
    wvioTbj.velocity.z = vwb.z();

    wvioTbj.acc_0.x = acc_0.x();
    wvioTbj.acc_0.y = acc_0.y();
    wvioTbj.acc_0.z = acc_0.z();

    wvioTbj.gyr_0.x = gyr_0.x();
    wvioTbj.gyr_0.y = gyr_0.y();
    wvioTbj.gyr_0.z = gyr_0.z();

    wvioTbj.BackTime = t0;

    poseij_imu.pose_j = wvioTbj.pose;
    poseij_imu.velocity_j = wvioTbj.velocity;
    poseij_imu.BackTime_j = wvioTbj.BackTime;
    poseij_imu.bias_acc_j = wvioTbj.bias_acc;
    poseij_imu.bias_gyro_j = wvioTbj.bias_gyro;
}

bool EstimatePredictPose(std::vector<Eigen::Vector3d> &imu_cur_data, add_msg::RelativePoseIMU& pij_imu){
    //estimte wvioTbody in laser of curr timestamp.
    //find the same time of fullPredictPoseBuf info.
    //wvioTbody_laser_time is wvioTbj
    pij_imu.BackTime_i = wvioTbi.BackTime;
    pij_imu.pose_i = wvioTbi.pose;
    pij_imu.velocity_i = wvioTbi.velocity;
    pij_imu.acc_0_i = wvioTbi.acc_0;
    pij_imu.gyr_0_i = wvioTbi.gyr_0;
    pij_imu.bias_acc_i = wvioTbi.bias_acc;
    pij_imu.bias_gyro_i = wvioTbi.bias_gyro;

    double close_time = 100;
    bool find_match_vpose = false;
    imu_cur_data.clear();

    mBuf_imu_preint.lock();
    //find the wvioTbody info which close to time LaserCloudFullRes
    add_msg::ImuPredict vpose_match;
    while(!fullPredictPoseBuf.empty()){
        add_msg::ImuPredictConstPtr vpose_msg = fullPredictPoseBuf.front();
        double vpose_time = vpose_msg->BackTime;
        if(vpose_time <= timeLaserCloudFullRes){
            find_match_vpose = true;
            if((timeLaserCloudFullRes - vpose_time) < close_time){
                close_time = (timeLaserCloudFullRes - vpose_time);
                vpose_match = *vpose_msg;
            }
            fullPredictPoseBuf.pop();
        }
        else{
            break;
        }
    }
    mBuf_imu_preint.unlock();

    if(find_match_vpose == false && !systemInited){
        std::cout << "timeLaserCloudFullRes=" << timeLaserCloudFullRes <<std::endl;
        std::cout << "vpose_msg=" << fullPredictPoseBuf.front()->BackTime <<std::endl;
        std::cout << "Initialize failure :: no matching between wvioTbody and LaserCloud" <<std::endl;
        ros::shutdown();
        return false;
    }
    else if(find_match_vpose == false && systemInited){
        //use last of visual imu pre-integation result for predict pose.
        vpose_match = wvioTbi;
    }

    if(vpose_match.BackTime == timeLaserCloudFullRes){
        //vpose of timeStamp = LaserCloud time stamp, we don't need to predict the pose.
        pij_imu.BackTime_j = -1;
        wvioTbj = vpose_match;
        return true;
    }
    else{
        //we need to predict wvioTbody in timestamp of LaserCloud.
        //Step1.get IMU raw info for predict pose.
        mBuf_imu_raw.lock();
        double lastimu_time = vpose_match.BackTime;
        double lastimu_Posei_time = -1;
        if(systemInited){
            lastimu_Posei_time = wvioTbi.BackTime;
        }
        std::vector<sensor_msgs::Imu> imu_msg_predict;

        //inser imu raw ij to pose_ij_imu
        while(!fullImuRawBuf.empty()){
            sensor_msgs::ImuConstPtr imu_msg = fullImuRawBuf.front();
            double imuraw_time = imu_msg->header.stamp.toSec();
            //inser raw imu info to pij_imu from
            //(wvioTi_back_time)*---(wvioTj_from_visual)*---(wvioTj_back_time)*
            if(imuraw_time > lastimu_time && imuraw_time <= timeLaserCloudFullRes){
                //for predict cur pose
                //(wvioTj_from_visual)*---(wvioTj_back_time)*
                imu_msg_predict.push_back(*(imu_msg));
                add_msg::Intervalimu imu_raw_tmp;
                imu_raw_tmp.time = imuraw_time;
                imu_raw_tmp.acc_x = imu_msg->linear_acceleration.x;
                imu_raw_tmp.acc_y = imu_msg->linear_acceleration.y;
                imu_raw_tmp.acc_z = imu_msg->linear_acceleration.z;
                imu_raw_tmp.gyr_x = imu_msg->angular_velocity.x;
                imu_raw_tmp.gyr_y = imu_msg->angular_velocity.y;
                imu_raw_tmp.gyr_z = imu_msg->angular_velocity.z;
                pij_imu.imu_raw_info.emplace_back(imu_raw_tmp);
                fullImuRawBuf.pop();
            }
            else if(imuraw_time <= lastimu_Posei_time && lastimu_Posei_time != -1){
                //*----(wvioTi_back_time)*
                fullImuRawBuf.pop();
            }
            else if(imuraw_time > lastimu_Posei_time && imuraw_time <= lastimu_time){
                //(wvioTi_back_time)*----(wvioTj_from_visual)*
                add_msg::Intervalimu imu_raw_tmp;
                imu_raw_tmp.time = imuraw_time;
                imu_raw_tmp.acc_x = imu_msg->linear_acceleration.x;
                imu_raw_tmp.acc_y = imu_msg->linear_acceleration.y;
                imu_raw_tmp.acc_z = imu_msg->linear_acceleration.z;
                imu_raw_tmp.gyr_x = imu_msg->angular_velocity.x;
                imu_raw_tmp.gyr_y = imu_msg->angular_velocity.y;
                imu_raw_tmp.gyr_z = imu_msg->angular_velocity.z;
                pij_imu.imu_raw_info.emplace_back(imu_raw_tmp);
                fullImuRawBuf.pop();
            }
            else if(imuraw_time > timeLaserCloudFullRes){
                break;
            }
        }
        mBuf_imu_raw.unlock();

        if(imu_msg_predict.size() < 3){
            ROS_ERROR_STREAM("only two imu_info for predict pose.");
        }

        //Step2.
        //Predict pose by imu_raw and wvioTbody
        ImuIntegrationForNextPose(vpose_match, pij_imu, imu_msg_predict);
        pij_imu.header = wvioTbj.header;
        pij_imu.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
        return true;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    printf("Mapping %d Hz \n", 10 / skipFrameNum);

    double qw, qx, qy, qz, tx, ty, tz;
    GetParam("/lidar_T_body/tx", tx);
    GetParam("/lidar_T_body/ty", ty);
    GetParam("/lidar_T_body/tz", tz);
    GetParam("/lidar_T_body/qx", qx);
    GetParam("/lidar_T_body/qy", qy);
    GetParam("/lidar_T_body/qz", qz);
    GetParam("/lidar_T_body/qw", qw);
    gw.x() = 0;
    gw.y() = 0;
    GetParam("/IMU_INFO/g_norm", gw.z());
    lidar_T_body.so3().setQuaternion(Eigen::Quaterniond(qw, qx, qy, qz));
    lidar_T_body.translation().x() = tx;
    lidar_T_body.translation().y() = ty;
    lidar_T_body.translation().z() = tz;

    //from cppt or other VIO output
    ros::Subscriber subVisualIMUInfo = nh.subscribe<add_msg::ImuPredict>("visual_imupreint_info", 1000, VisualImuInfoHandler);
    //from imu sensor
    ros::Subscriber subRawImuFullInfo = nh.subscribe<sensor_msgs::Imu>("/imu/data_raw", 1000, ImuRawInfoHandler);

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_CornerPointsLessSharp(nh, "/laser_cloud_less_sharp", 200);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_SurfPointsLessFlat(nh, "/laser_cloud_less_flat", 200);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_LaserCloudFullRes(nh, "/velodyne_cloud_2", 200);
    typedef  message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2 ,
            sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_CornerPointsLessSharp,
                                                     sub_SurfPointsLessFlat, sub_LaserCloudFullRes);
    sync.registerCallback(boost::bind(&Sync_callback, _1, _2, _3));

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    ros::Publisher pubPoseIJ_rawIMU = nh.advertise<add_msg::RelativePoseIMU>("/Poseij_imu", 100);

    nav_msgs::Path laserPath;

    int frameCount = 0;
    //using 100hz be a circle to implement while loop
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();

        if (!cornerLessSharpBuf.empty() && !surfLessFlatBuf.empty() &&
                !fullPointsBuf.empty() && !fullPredictPoseBuf.empty())
        {
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            if (timeCornerPointsLessSharp != timeLaserCloudFullRes || timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync lidar messeage!");
                ROS_BREAK();
            }

            mBuf.lock();
            if (1)
            {
                sensor_msgs::PointCloud2 laserCloudCornerLast2 = *cornerLessSharpBuf.front();
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudCornerLast2.header.frame_id = "/world";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);
                cornerLessSharpBuf.pop();

                sensor_msgs::PointCloud2 laserCloudSurfLast2 = *surfLessFlatBuf.front();
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudSurfLast2.header.frame_id = "/world";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
                surfLessFlatBuf.pop();

                sensor_msgs::PointCloud2 laserCloudFullRes3 =*fullPointsBuf.front();
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudFullRes3.header.frame_id = "/world";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
                fullPointsBuf.pop();
            }
            mBuf.unlock();

            std::vector<Eigen::Vector3d> imu_cur_data;
            add_msg::RelativePoseIMU pij_imu;
            pij_imu.BackTime_i = -1;
            pij_imu.BackTime_j = -1;
            bool findMatchVisualPose = EstimatePredictPose(imu_cur_data, pij_imu);
            TicToc t_whole;
            // initializing
            if (!systemInited && findMatchVisualPose)
            {
                pij_imu.BackTime_i = -1;
                pij_imu.BackTime_j = -1;
                systemInited = true;
                wvioTbi = wvioTbj;
                para_R.setQuaternion(Eigen::Quaterniond::Identity());
                q_w_curr.setQuaternion(Eigen::Quaterniond::Identity());
                para_P.setZero();
                para_pose = new double[7];
                std::cout << "Initialization finished \n";
            }
            else if(systemInited)
            {

                Sophus::SE3d lidari_T_lidarj;
                //bodyi_bodyj
                Eigen::Quaterniond wvio_Qbi;
                wvio_Qbi.x() = wvioTbi.pose.orientation.x;
                wvio_Qbi.y() = wvioTbi.pose.orientation.y;
                wvio_Qbi.z() = wvioTbi.pose.orientation.z;
                wvio_Qbi.w() = wvioTbi.pose.orientation.w;
                Eigen::Vector3d wvio_tbi(wvioTbi.pose.position.x, wvioTbi.pose.position.y, wvioTbi.pose.position.z);
                Eigen::Quaterniond wvio_Qbj;
                wvio_Qbj.x() = wvioTbj.pose.orientation.x;
                wvio_Qbj.y() = wvioTbj.pose.orientation.y;
                wvio_Qbj.z() = wvioTbj.pose.orientation.z;
                wvio_Qbj.w() = wvioTbj.pose.orientation.w;
                Eigen::Vector3d wvio_tbj(wvioTbj.pose.position.x, wvioTbj.pose.position.y, wvioTbj.pose.position.z);
                Sophus::SE3d T_wvioTbi(wvio_Qbi, wvio_tbi);
                Sophus::SE3d T_wvioTbj(wvio_Qbj, wvio_tbj);
                Sophus::SE3d body_T_lidar = lidar_T_body.inverse();
                lidari_T_lidarj = (T_wvioTbi * body_T_lidar).inverse() * (T_wvioTbj * body_T_lidar);
                //Sophus::SE3d lidari_T_lidarj = (wvioTbi ).inverse() * (wvioTbj);


                //debug
                //std::cout << "T_wvioTbi t=" << T_wvioTbi.translation() << "_q=" << T_wvioTbi.unit_quaternion().coeffs() <<std::endl;
                //std::cout << "T_wvioTbi v=" << Eigen::Vector3d(wvioTbi.velocity.x, wvioTbi.velocity.y, wvioTbi.velocity.z) <<std::endl;
                //std::cout << "T_wvioTbj t=" << T_wvioTbj.translation() << "_q=" << T_wvioTbj.unit_quaternion().coeffs() <<std::endl;
                //std::cout << "T_wvioTbj v=" << Eigen::Vector3d(wvioTbj.velocity.x, wvioTbj.velocity.y, wvioTbj.velocity.z) <<std::endl;
                //for(int imuid = 0; imuid < pij_imu.imu_raw_info.size(); imuid++){
                //
                //}
                //debug

                wvioTbi = wvioTbj;
                para_R = lidari_T_lidarj.so3();
                para_P = lidari_T_lidarj.translation();
                t_w_curr = t_w_curr + (q_w_curr * para_P);
                q_w_curr = q_w_curr * para_R;
            }


            //Publish poseij and raw imu info for optimize bias in laserMapping
            pubPoseIJ_rawIMU.publish(pij_imu);
            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/world";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserOdometry.pose.pose.orientation.x = q_w_curr.unit_quaternion().x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.unit_quaternion().y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.unit_quaternion().z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.unit_quaternion().w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/world";
            pubLaserPath.publish(laserPath);

            //printf("publication time %f ms \n", t_pub.toc());
            //printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;
        }
        rate.sleep();
    }
    return 0;
}
