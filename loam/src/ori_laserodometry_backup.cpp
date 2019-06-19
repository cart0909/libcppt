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
#include "vloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"
#include "sophus/se3.hpp"
#include "ceres/local_parameterization_se3.h"
#include "add_msg/ImuPredict.h"
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


void Sync_callback(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2,
                   const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2,
                   const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2,
                   const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2,
                   const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2){
        mBuf.lock();
        cornerSharpBuf.push(cornerPointsSharp2);
        cornerLessSharpBuf.push(cornerPointsLessSharp2);
        surfFlatBuf.push(surfPointsFlat2);
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

void ImuIntegrationForNextPose(add_msg::ImuPredict& vpose_match, std::vector<sensor_msgs::Imu> imu_msg_predict)
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
    Eigen::Vector3d gw(0, 0, 9.81007);


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
}

bool EstimatePredictPose(){
    //estimte wvioTbody in laser of curr timestamp.
    //find the same time of fullPredictPoseBuf info.
    //wvioTbody_laser_time is wvioTbj
    double close_time = 100;
    bool find_match_vpose = false;

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
        std::cout << "cannot find any matching between wvioTbody and LaserCloud" <<std::endl;
        return false;
    }
    else if(find_match_vpose == false && systemInited){
        vpose_match = wvioTbi;
    }

    if(vpose_match.BackTime == timeLaserCloudFullRes){
        //vpose of timeStamp = LaserCloud time stamp, we don't need to predict the pose.
        wvioTbj = vpose_match;
        return true;
    }
    else{
        //we need to predict wvioTbody in timestamp of LaserCloud.
        //Step1.get IMU raw info for predict pose.
        mBuf_imu_raw.lock();
        double lastimu_time = vpose_match.BackTime;
        std::vector<sensor_msgs::Imu> imu_msg_predict;
        while(!fullImuRawBuf.empty()){
            sensor_msgs::ImuConstPtr imu_msg = fullImuRawBuf.front();
            double imuraw_time = imu_msg->header.stamp.toSec();
            if(imuraw_time > lastimu_time && imuraw_time <= timeLaserCloudFullRes){
                imu_msg_predict.push_back(*(imu_msg));
                fullImuRawBuf.pop();
            }
            else if(imuraw_time <= lastimu_time){
                fullImuRawBuf.pop();
            }
            else if(imuraw_time > timeLaserCloudFullRes){
                break;
            }
        }
        mBuf_imu_raw.unlock();

        //Step2.
        //Predict pose by imu_raw and wvioTbody
        ImuIntegrationForNextPose(vpose_match, imu_msg_predict);

        return true;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    printf("Mapping %d Hz \n", 10 / skipFrameNum);

    double tx, ty, tz, qx, qy, qz, qw;
    //
    nh.param<double>("tx", tx, 0);
    nh.param<double>("ty", ty, 0);
    nh.param<double>("tz", tz, 0);
    nh.param<double>("qx", qx, 0);
    nh.param<double>("qy", qy, 0);
    nh.param<double>("qz", qz, 0);
    nh.param<double>("qw", qw, 0);
    lidar_T_body.so3().setQuaternion(Eigen::Quaterniond(qw, qx, qy, qz));
    lidar_T_body.translation().x() = tx;
    lidar_T_body.translation().y() = ty;
    lidar_T_body.translation().z() = tz;

    //from cppt or other VIO output
    ros::Subscriber subVisualIMUInfo = nh.subscribe<add_msg::ImuPredict>("visual_imupreint_info", 1000, VisualImuInfoHandler);
    //from imu sensor
    ros::Subscriber subRawImuFullInfo = nh.subscribe<sensor_msgs::Imu>("/imu/data_raw", 1000, ImuRawInfoHandler);

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_CornerPointsSharp(nh, "/laser_cloud_sharp", 200);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_CornerPointsLessSharp(nh, "/laser_cloud_less_sharp", 200);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_SurfPointsFlat(nh, "/laser_cloud_less_sharp", 200);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_SurfPointsLessFlat(nh, "/laser_cloud_less_flat", 200);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_LaserCloudFullRes(nh, "/velodyne_cloud_2", 200);
    typedef  message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2 ,
            sensor_msgs::PointCloud2 , sensor_msgs::PointCloud2 , sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_CornerPointsSharp, sub_CornerPointsLessSharp, sub_SurfPointsFlat,
                                                     sub_SurfPointsLessFlat, sub_LaserCloudFullRes);
    sync.registerCallback(boost::bind(&Sync_callback, _1, _2, _3, _4, _5));

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;

    int frameCount = 0;
    //using 100hz be a circle to implement while loop
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();

        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
                !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
                !fullPointsBuf.empty() && !fullPredictPoseBuf.empty())
        {
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                    timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                    timeSurfPointsFlat != timeLaserCloudFullRes ||
                    timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync lidar messeage!");
                ROS_BREAK();
            }

            mBuf.lock();
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();

            bool findMatchVisualPose = EstimatePredictPose();
            TicToc t_whole;
            // initializing
            if (!systemInited && findMatchVisualPose)
            {
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
#if 1
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
                wvioTbi = wvioTbj;
                para_R = lidari_T_lidarj.so3();
                para_P = lidari_T_lidarj.translation();

#else
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                int surfPointsFlatNum = surfPointsFlat->points.size();

                TicToc t_opt;
                std::memcpy(para_pose , para_R.data(), sizeof(double) * Sophus::SO3d::num_parameters);
                std::memcpy(para_pose + 4, para_P.data(), sizeof(double) * 3);
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    //ceres::LossFunction *loss_function = NULL;
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LocalParameterization *local_para_se3 = new autodiff::LocalParameterizationSE3();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);
                    //set Pose
                    problem.AddParameterBlock(para_pose, 7, local_para_se3);
                    pcl::PointXYZI pointSel;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;
                    TicToc t_data;
                    // find correspondence for corner features
                    // for sharp point
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                        //Initialize, kdtreeCornerLast is null return pointSearchInd = 0
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                        int closestPointInd = -1, minPointInd2 = -1;
                        //step1. find the line that means to find mid / cloest of two points.
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) //the closestPoint distance need less than 25
                        {
                            closestPointInd = pointSearchInd[0];
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                        (laserCloudCornerLast->points[j].y - pointSel.y) *
                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                        (laserCloudCornerLast->points[j].z - pointSel.z) *
                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                        (laserCloudCornerLast->points[j].y - pointSel.y) *
                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                        (laserCloudCornerLast->points[j].z - pointSel.z) *
                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }

                        //step2. if find the mid and closest point.
                        //insert to cost function, 1.cur_point, 2.cloest_point 3. mid_point 4.realtime(s)
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            double s;
                            if (DISTORTION)
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;
                            ceres::CostFunction *cost_function = LidarEdgeFactorSO3::Create(curr_point, last_point_a, last_point_b, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_pose);
                            corner_correspondence++;
                        }
                    }

                    // find correspondence for plane features
                    // for plane point
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];

                            // get closest point's scan ID
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                        (laserCloudSurfLast->points[j].y - pointSel.y) *
                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                        (laserCloudSurfLast->points[j].z - pointSel.z) *
                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or lower scan line
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                        (laserCloudSurfLast->points[j].y - pointSel.y) *
                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                        (laserCloudSurfLast->points[j].z - pointSel.z) *
                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                           surfPointsFlat->points[i].y,
                                                           surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                             laserCloudSurfLast->points[closestPointInd].y,
                                                             laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                             laserCloudSurfLast->points[minPointInd2].y,
                                                             laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                             laserCloudSurfLast->points[minPointInd3].y,
                                                             laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;
                                ceres::CostFunction *cost_function = LidarPlaneFactorSO3::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                problem.AddResidualBlock(cost_function, loss_function, para_pose);
                                plane_correspondence++;
                            }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    //printf("data association time %f ms \n", t_data.toc());

                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }

                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    //printf("solver time %f ms \n", t_solver.toc());
                }
                //printf("optimization twice time %f \n", t_opt.toc());
                //get new pose
                std::memcpy(para_R.data(), para_pose , sizeof(double) * Sophus::SO3d::num_parameters);
                std::memcpy(para_P.data(), para_pose + 4, sizeof (double) * 3);
#endif
                t_w_curr = t_w_curr + (q_w_curr * para_P);
                q_w_curr = q_w_curr * para_R;
            }

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

            //TODO::Using visual IMU predict this pose.
            // transform corner features and plane features to the scan end point
            if (0)
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }

            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';

            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            if (1)
            {
                frameCount = 0;

                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudCornerLast2.header.frame_id = "/world";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudSurfLast2.header.frame_id = "/world";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudFullRes3.header.frame_id = "/world";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
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
