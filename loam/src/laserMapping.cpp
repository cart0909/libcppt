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
#include "laserMapping.h"

#define PUB_ORI_PCL2 0
int frameCount = 0;
double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;
double timeOrigionalCloud = 0;
double timePoseijImuLast = 0 ;
int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;


const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851

int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<PointType>::Ptr OrigionalCloudLast(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// points in every cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> origional_cloud_Buf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<add_msg::RelativePoseIMUConstPtr> poseijImuBuf;
std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;
ros::Publisher pub_TF_stamped, pubLaserCloudvoxblox;
nav_msgs::Path laserAfterMappedPath;


// set initial guess
void transformAssociateToMap()
{
    Sophus::SE3d wmapTodom(q_wmap_wodom, t_wmap_wodom);
    Sophus::SE3d wodomTcurr(q_wodom_curr, t_wodom_curr);
    Sophus::SE3d wmapTlidar = wmapTodom *  wodomTcurr * lidar_T_body.inverse();
    q_w_lidarj = wmapTlidar.so3().unit_quaternion();
    t_w_lidarj = wmapTlidar.translation();
}

void transformUpdate()
{
    Sophus::SE3d wmapTlidar(q_w_lidarj, t_w_lidarj);
    Sophus::SE3d wodomTbody_cur(q_wodom_curr, t_wodom_curr);
    Sophus::SE3d wmapTodom = wmapTlidar * lidar_T_body * wodomTbody_cur.inverse();

    q_wmap_wodom = wmapTodom.so3().unit_quaternion();
    t_wmap_wodom = wmapTodom.translation();
}

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_lidarj * point_curr + t_w_lidarj;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->intensity = pi->intensity;
    //po->intensity = 1.0;
}

void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_curr = q_w_lidarj.inverse() * (point_w - t_w_lidarj);
    po->x = point_curr.x();
    po->y = point_curr.y();
    po->z = point_curr.z();
    po->intensity = pi->intensity;
}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2, const sensor_msgs::PointCloud2ConstPtr &Origional_point_cloud)
{
    mBuf.lock();
    cornerLastBuf.push(laserCloudCornerLast2);
    origional_cloud_Buf.push(Origional_point_cloud);
    mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
    mBuf.lock();
    surfLastBuf.push(laserCloudSurfLast2);
    mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullResBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}



void relativePoseIMUHandler(const add_msg::RelativePoseIMUConstPtr &poseij_imu)
{
    //    std::cout << "===============laserMapping=========" <<std::endl;
    //        Eigen::Quaterniond wvio_Qbi;
    //        wvio_Qbi.x() = poseij_imu->pose_i.orientation.x;
    //        wvio_Qbi.y() = poseij_imu->pose_i.orientation.y;
    //        wvio_Qbi.z() = poseij_imu->pose_i.orientation.z;
    //        wvio_Qbi.w() = poseij_imu->pose_i.orientation.w;
    //        Eigen::Vector3d wvio_tbi(poseij_imu->pose_i.position.x, poseij_imu->pose_i.position.y, poseij_imu->pose_i.position.z);
    //        Eigen::Quaterniond wvio_Qbj;
    //        wvio_Qbj.x() = poseij_imu->pose_j.orientation.x;
    //        wvio_Qbj.y() = poseij_imu->pose_j.orientation.y;
    //        wvio_Qbj.z() = poseij_imu->pose_j.orientation.z;
    //        wvio_Qbj.w() = poseij_imu->pose_j.orientation.w;
    //        Eigen::Vector3d wvio_tbj(poseij_imu->pose_j.position.x, poseij_imu->pose_j.position.y, poseij_imu->pose_j.position.z);
    //        Sophus::SE3d T_wvioTbi(wvio_Qbi, wvio_tbi);
    //        Sophus::SE3d T_wvioTbj(wvio_Qbj, wvio_tbj);

    //        std::cout << "T_wvioTbi t=" << T_wvioTbi.translation() << "_q=" << T_wvioTbi.unit_quaternion().coeffs() <<std::endl;
    //        std::cout << "T_wvioTbi v=" << Eigen::Vector3d(poseij_imu->velocity_i.x, poseij_imu->velocity_i.y, poseij_imu->velocity_i.z) <<std::endl;
    //        std::cout << "T_wvioTbj t=" << T_wvioTbj.translation() << "_q=" << T_wvioTbj.unit_quaternion().coeffs() <<std::endl;
    //        std::cout << "T_wvioTbj v=" << Eigen::Vector3d(poseij_imu->velocity_j.x, poseij_imu->velocity_j.y, poseij_imu->velocity_j.z) <<std::endl;
    //        std::cout << "PoseI time=" << poseij_imu->BackTime_i <<std::endl;
    //        std::cout << "PoseJ time=" << poseij_imu->BackTime_j <<std::endl;
    //        for(int imuid = 0; imuid < poseij_imu->imu_raw_info.size(); imuid++){
    //            std::cout << "imu_id=" << imuid << "_time=" << poseij_imu->imu_raw_info[imuid].time <<std::endl;
    //            std::cout << "imuid=" << imuid << "_acc=" << Eigen::Vector3d(poseij_imu->imu_raw_info[imuid].gyr_x, poseij_imu->imu_raw_info[imuid].gyr_y, poseij_imu->imu_raw_info[imuid].gyr_z) <<std::endl;
    //       }
    mBuf.lock();
    poseijImuBuf.push(poseij_imu);
    mBuf.unlock();
}

//receive odomtry
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
    mBuf.lock();
    odometryBuf.push(laserOdometry);
    mBuf.unlock();

    // high frequence publish
    Eigen::Quaterniond tmp_q_wodom_curr;
    Eigen::Vector3d tmp_t_wodom_curr;
    tmp_q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
    tmp_q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
    tmp_q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
    tmp_q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
    tmp_t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
    tmp_t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
    tmp_t_wodom_curr.z() = laserOdometry->pose.pose.position.z;
    Sophus::SE3d odomTbody(tmp_q_wodom_curr, tmp_t_wodom_curr);
    Sophus::SE3d wmapTwodom(q_wmap_wodom, t_wmap_wodom);
    Sophus::SE3d wmapTlidar = wmapTwodom * odomTbody * lidar_T_body.inverse();

    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = "/world";
    odomAftMapped.child_frame_id = "/aft_mapped";
    odomAftMapped.header.stamp = laserOdometry->header.stamp;
    odomAftMapped.pose.pose.orientation.x = wmapTlidar.so3().unit_quaternion().x();
    odomAftMapped.pose.pose.orientation.y = wmapTlidar.so3().unit_quaternion().y();
    odomAftMapped.pose.pose.orientation.z = wmapTlidar.so3().unit_quaternion().z();
    odomAftMapped.pose.pose.orientation.w = wmapTlidar.so3().unit_quaternion().w();
    odomAftMapped.pose.pose.position.x = wmapTlidar.translation().x();
    odomAftMapped.pose.pose.position.y = wmapTlidar.translation().y();
    odomAftMapped.pose.pose.position.z = wmapTlidar.translation().z();
    pubOdomAftMappedHighFrec.publish(odomAftMapped);
}

void PushPoseIJTolidarState(LidarStatePtr lidarState, add_msg::RelativePoseIMU& poseij_imu){
    //InsertSomeInfo from RelativePoseIMU to LidarState
    //without 14.lidarState->v_acc 15. lidarState->v_gyr
    // 16.lidarState->imupreinte 17.lidarState->exist_imu

    //1.
    lidarState->BackTime_i = poseij_imu.BackTime_i;
    //2.
    lidarState->BackTime_j = poseij_imu.BackTime_j;
    //3.
    lidarState->lidar_time = poseij_imu.header.stamp.toSec();
    //4.
    lidarState->bias_acc_i.x() = poseij_imu.bias_acc_i.x;
    lidarState->bias_acc_i.y() = poseij_imu.bias_acc_i.y;
    lidarState->bias_acc_i.z() = poseij_imu.bias_acc_i.z;
    //5.
    lidarState->bias_gyr_i.x() = poseij_imu.bias_gyro_i.x;
    lidarState->bias_gyr_i.y() = poseij_imu.bias_gyro_i.y;
    lidarState->bias_gyr_i.z() = poseij_imu.bias_gyro_i.z;
    //6.
    lidarState->bias_acc_j.x() = poseij_imu.bias_acc_j.x;
    lidarState->bias_acc_j.y() = poseij_imu.bias_acc_j.y;
    lidarState->bias_acc_j.z() = poseij_imu.bias_acc_j.z;
    //7.
    lidarState->bias_gyr_j.x() = poseij_imu.bias_gyro_j.x;
    lidarState->bias_gyr_j.y() = poseij_imu.bias_gyro_j.y;
    lidarState->bias_gyr_j.z() = poseij_imu.bias_gyro_j.z;

    //8.wmap_q_lidari: poseij_imu is wodom_T_body
    Eigen::Quaterniond wodom_Qbi;
    wodom_Qbi.x() = poseij_imu.pose_i.orientation.x;
    wodom_Qbi.y() = poseij_imu.pose_i.orientation.y;
    wodom_Qbi.z() = poseij_imu.pose_i.orientation.z;
    wodom_Qbi.w() = poseij_imu.pose_i.orientation.w;
    //convert to wmap
    lidarState->wmap_Qbi.setQuaternion(q_wmap_wodom * wodom_Qbi);

    //9.wmap_t_lidari:
    lidarState->wmap_tbi.x() = poseij_imu.pose_i.position.x;
    lidarState->wmap_tbi.y() = poseij_imu.pose_i.position.y;
    lidarState->wmap_tbi.z() = poseij_imu.pose_i.position.z;
    lidarState->wmap_tbi = q_wmap_wodom  * lidarState->wmap_tbi + t_wmap_wodom;

    //10.wamp_velocity_lidari
    lidarState->wmap_veci.x() = poseij_imu.velocity_i.x;
    lidarState->wmap_veci.y() = poseij_imu.velocity_i.y;
    lidarState->wmap_veci.z() = poseij_imu.velocity_i.z;
    lidarState->wmap_veci = q_wmap_wodom * lidarState->wmap_veci;

    //11.wmap_q_lidarj: poseij_imu is wodom_T_body
    Eigen::Quaterniond wodom_Qbj;
    wodom_Qbj.x() = poseij_imu.pose_j.orientation.x;
    wodom_Qbj.y() = poseij_imu.pose_j.orientation.y;
    wodom_Qbj.z() = poseij_imu.pose_j.orientation.z;
    wodom_Qbj.w() = poseij_imu.pose_j.orientation.w;
    //convert to wmap
    lidarState->wmap_Qbj.setQuaternion(q_wmap_wodom * wodom_Qbj);

    //12.wmap_t_lidarj:j
    lidarState->wmap_tbj.x() = poseij_imu.pose_j.position.x;
    lidarState->wmap_tbj.y() = poseij_imu.pose_j.position.y;
    lidarState->wmap_tbj.z() = poseij_imu.pose_j.position.z;
    lidarState->wmap_tbj = q_wmap_wodom  * lidarState->wmap_tbj + t_wmap_wodom;

    //13.wamp_velocity_lidarj
    lidarState->wmap_vecj.x() = poseij_imu.velocity_j.x;
    lidarState->wmap_vecj.y() = poseij_imu.velocity_j.y;
    lidarState->wmap_vecj.z() = poseij_imu.velocity_j.z;
    lidarState->wmap_vecj = q_wmap_wodom * lidarState->wmap_vecj;
}

void process()
{
    while(1)
    {
        while (!cornerLastBuf.empty() && !surfLastBuf.empty() &&
               !fullResBuf.empty() && !odometryBuf.empty() && !origional_cloud_Buf.empty() && !poseijImuBuf.empty())
        {
            mBuf.lock();
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                odometryBuf.pop();
            if (odometryBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                surfLastBuf.pop();
            if (surfLastBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                fullResBuf.pop();
            if (fullResBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            while (!poseijImuBuf.empty() && poseijImuBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                poseijImuBuf.pop();
            if (poseijImuBuf.empty())
            {
                mBuf.unlock();
                break;
            }


            timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
            timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timeOrigionalCloud = origional_cloud_Buf.front()->header.stamp.toSec();
            timePoseijImuLast = poseijImuBuf.front()->header.stamp.toSec();
            if (timeLaserCloudCornerLast != timeLaserOdometry ||
                    timeLaserCloudSurfLast != timeLaserOdometry ||
                    timeLaserCloudFullRes != timeLaserOdometry ||
                    timeOrigionalCloud != timeLaserOdometry ||
                    timePoseijImuLast != timeLaserOdometry)
            {
                printf("time corner %f surf %f full %f odom %f orgional %f\n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry, timeOrigionalCloud);
                printf("unsync messeage!");
                mBuf.unlock();
                break;
            }

            laserCloudCornerLast->clear();
            pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
            cornerLastBuf.pop();

            laserCloudSurfLast->clear();
            pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
            surfLastBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
            fullResBuf.pop();

#if PUB_ORI_PCL2
            sensor_msgs::PointCloud2 laserCloudFullRes_voxblox;
            laserCloudFullRes_voxblox = (*origional_cloud_Buf.front());
            laserCloudFullRes_voxblox.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            pubLaserCloudvoxblox.publish(laserCloudFullRes_voxblox);
#endif
            origional_cloud_Buf.pop(); //no-scruibe

            q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
            q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
            q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
            q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
            t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
            t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
            t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
            odometryBuf.pop();

            //estimate imu-preintegration

            add_msg::RelativePoseIMU poseij_imu = *(poseijImuBuf.front());
            LidarStatePtr lidarState = std::make_shared<LidarState>();
            if(poseij_imu.BackTime_i != -1 && poseij_imu.BackTime_j != -1)
            {
                Eigen::Vector3d average_ba, average_bg, acc_0_i, gyr_0_i;
                //average_ba.x() = (poseij_imu.bias_acc_i.x + poseij_imu.bias_acc_j.x) * 0.5;
                //average_ba.y() = (poseij_imu.bias_acc_i.y + poseij_imu.bias_acc_j.y) * 0.5;
                //average_ba.z() = (poseij_imu.bias_acc_i.z + poseij_imu.bias_acc_j.z) * 0.5;
                //average_bg.x() = (poseij_imu.bias_gyro_i.x + poseij_imu.bias_gyro_j.x) * 0.5;
                //average_bg.y() = (poseij_imu.bias_gyro_i.y + poseij_imu.bias_gyro_j.y) * 0.5;
                //average_bg.z() = (poseij_imu.bias_gyro_i.z + poseij_imu.bias_gyro_j.z) * 0.5;

                average_ba.x() = (poseij_imu.bias_acc_i.x );
                average_ba.y() = (poseij_imu.bias_acc_i.y );
                average_ba.z() = (poseij_imu.bias_acc_i.z );
                average_bg.x() = (poseij_imu.bias_gyro_i.x);
                average_bg.y() = (poseij_imu.bias_gyro_i.y);
                average_bg.z() = (poseij_imu.bias_gyro_i.z);

                acc_0_i.x() = poseij_imu.acc_0_i.x;
                acc_0_i.y() = poseij_imu.acc_0_i.y;
                acc_0_i.z() = poseij_imu.acc_0_i.z;

                gyr_0_i.x() = poseij_imu.gyr_0_i.x;
                gyr_0_i.y() = poseij_imu.gyr_0_i.y;
                gyr_0_i.z() = poseij_imu.gyr_0_i.z;
                //Init imu preintegration
                lidarState->imupreinte = std::make_shared<IntegrationBase>(acc_0_i, gyr_0_i,
                                                                           average_ba, average_bg, acc_n, gyr_n,
                                                                           acc_w, gyr_w);

                PushPoseIJTolidarState(lidarState, poseij_imu);

                double t0 = poseij_imu.BackTime_i;
                Eigen::Vector3d acc_raw;
                Eigen::Vector3d gyr_raw;

                for(size_t r_id = 0; r_id < poseij_imu.imu_raw_info.size(); r_id++){
                    double t = poseij_imu.imu_raw_info[r_id].time, dt = t - t0;
                    acc_raw.x() = poseij_imu.imu_raw_info[r_id].acc_x;
                    acc_raw.y() = poseij_imu.imu_raw_info[r_id].acc_y;
                    acc_raw.z() = poseij_imu.imu_raw_info[r_id].acc_z;
                    gyr_raw.x() = poseij_imu.imu_raw_info[r_id].gyr_x;
                    gyr_raw.y() = poseij_imu.imu_raw_info[r_id].gyr_y;
                    gyr_raw.z() = poseij_imu.imu_raw_info[r_id].gyr_z;
                    lidarState->v_acc.emplace_back(acc_raw);
                    lidarState->v_gyr.emplace_back(gyr_raw);
                    lidarState->imupreinte->push_back(dt, acc_raw, gyr_raw);
                    t0 = t;
                }

                if(poseij_imu.imu_raw_info.size() > 5 && poseij_imu.imu_raw_info.size() < 30)
                    lidarState->exist_imu = true;
                //lidarState->exist_imu = false;
            }

            poseijImuBuf.pop();
            while(!cornerLastBuf.empty())
            {
                cornerLastBuf.pop();
                origional_cloud_Buf.pop();
                printf("drop lidar frame in mapping for real time performance \n");
            }
            mBuf.unlock();

            TicToc t_whole;
            if(!systemInited){
                //system initialize
                systemInited = true;
                Eigen::Quaterniond I;
                I.setIdentity();
                w_T_lidarj_last.so3().setQuaternion(I);
            }

            //:convert to wvioTlidar to wmapTlidarj by wmapTwvio
            transformAssociateToMap();
            Sophus::SE3d wmap_T_lidarj(q_w_lidarj, t_w_lidarj);
            w_T_bodyj = wmap_T_lidarj * lidar_T_body;

            TicToc t_shift;
            int centerCubeI = int((t_w_lidarj.x() + 25.0) / 50.0) + laserCloudCenWidth;
            int centerCubeJ = int((t_w_lidarj.y() + 25.0) / 50.0) + laserCloudCenHeight;
            int centerCubeK = int((t_w_lidarj.z() + 25.0) / 50.0) + laserCloudCenDepth;
            //+++shift map according to translation of wmap_T_lidarj
            if (t_w_lidarj.x() + 25.0 < 0)
                centerCubeI--;
            if (t_w_lidarj.y() + 25.0 < 0)
                centerCubeJ--;
            if (t_w_lidarj.z() + 25.0 < 0)
                centerCubeK--;

            while (centerCubeI < 3)
            {
                for (int j = 0; j < laserCloudHeight; j++)
                {
                    for (int k = 0; k < laserCloudDepth; k++)
                    {
                        int i = laserCloudWidth - 1;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        for (; i >= 1; i--)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeSurfPointer;
                        laserCloudCubeCornerPointer->clear();
                        laserCloudCubeSurfPointer->clear();
                    }
                }

                centerCubeI++;
                laserCloudCenWidth++;
            }

            while (centerCubeI >= laserCloudWidth - 3)
            {
                for (int j = 0; j < laserCloudHeight; j++)
                {
                    for (int k = 0; k < laserCloudDepth; k++)
                    {
                        int i = 0;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        for (; i < laserCloudWidth - 1; i++)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeSurfPointer;
                        laserCloudCubeCornerPointer->clear();
                        laserCloudCubeSurfPointer->clear();
                    }
                }

                centerCubeI--;
                laserCloudCenWidth--;
            }

            while (centerCubeJ < 3)
            {
                for (int i = 0; i < laserCloudWidth; i++)
                {
                    for (int k = 0; k < laserCloudDepth; k++)
                    {
                        int j = laserCloudHeight - 1;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        for (; j >= 1; j--)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeSurfPointer;
                        laserCloudCubeCornerPointer->clear();
                        laserCloudCubeSurfPointer->clear();
                    }
                }

                centerCubeJ++;
                laserCloudCenHeight++;
            }

            while (centerCubeJ >= laserCloudHeight - 3)
            {
                for (int i = 0; i < laserCloudWidth; i++)
                {
                    for (int k = 0; k < laserCloudDepth; k++)
                    {
                        int j = 0;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        for (; j < laserCloudHeight - 1; j++)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeSurfPointer;
                        laserCloudCubeCornerPointer->clear();
                        laserCloudCubeSurfPointer->clear();
                    }
                }

                centerCubeJ--;
                laserCloudCenHeight--;
            }

            while (centerCubeK < 3)
            {
                for (int i = 0; i < laserCloudWidth; i++)
                {
                    for (int j = 0; j < laserCloudHeight; j++)
                    {
                        int k = laserCloudDepth - 1;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        for (; k >= 1; k--)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeSurfPointer;
                        laserCloudCubeCornerPointer->clear();
                        laserCloudCubeSurfPointer->clear();
                    }
                }

                centerCubeK++;
                laserCloudCenDepth++;
            }

            while (centerCubeK >= laserCloudDepth - 3)
            {
                for (int i = 0; i < laserCloudWidth; i++)
                {
                    for (int j = 0; j < laserCloudHeight; j++)
                    {
                        int k = 0;
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        for (; k < laserCloudDepth - 1; k++)
                        {
                            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                    laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
                        }
                        laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeCornerPointer;
                        laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                laserCloudCubeSurfPointer;
                        laserCloudCubeCornerPointer->clear();
                        laserCloudCubeSurfPointer->clear();
                    }
                }

                centerCubeK--;
                laserCloudCenDepth--;
            }

            //---shift map according to translation of wmap_T_lidarj

            int laserCloudValidNum = 0;
            int laserCloudSurroundNum = 0;

            //+++select the region(5*5*3) of map for curr 3d point matching
            for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
            {
                for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
                {
                    for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
                    {
                        if (i >= 0 && i < laserCloudWidth &&
                                j >= 0 && j < laserCloudHeight &&
                                k >= 0 && k < laserCloudDepth)
                        {
                            //k is depth, i is width, j is height of map.
                            //laserCloudValidInd, laserCloudSurroundInd save index for map laserCloudCornerArray, laserCloudSurfArray
                            laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                            laserCloudValidNum++;
                            laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                            laserCloudSurroundNum++;
                        }
                    }
                }
            }

            laserCloudCornerFromMap->clear();
            laserCloudSurfFromMap->clear();
            for (int i = 0; i < laserCloudValidNum; i++)
            {
                //Extract region of map from laserCloudCornerArray to laserCloudCornerFromMap.
                //which relate to the ValidInd(from cube IJK)
                *laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
                *laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
            }
            int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
            int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();
            //---select the region(5*5*3) of map for curr 3d point matching

            //Down sample 3D point at curr time.
            pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
            downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
            downSizeFilterCorner.filter(*laserCloudCornerStack);
            int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

            pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
            downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
            downSizeFilterSurf.filter(*laserCloudSurfStack);
            int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

            printf("map prepare time %f ms\n", t_shift.toc());
            printf("map corner num %d  surf num %d \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);
            if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50)
            {
                TicToc t_opt;
                TicToc t_tree;
                kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
                kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
                printf("build tree time %f ms \n", t_tree.toc());

                //+++Optimize laserMapping do twice
                for (int iterCount = 0; iterCount < 2; iterCount++)
                {
                    //ceres::LossFunction *loss_function = NULL;
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    //ceres::LocalParameterization *q_parameterization =
                    //        new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;


                    ceres::Problem problem(problem_options);
                    ceres::LocalParameterization *local_para_se3 = new LocalParameterizationSE3();
                    problem.AddParameterBlock(parameters_j, 7, local_para_se3);
                    //problem.AddParameterBlock(parameters_j, 4, q_parameterization);
                    //problem.AddParameterBlock(parameters_j + 4, 3);

                    TicToc t_data;
                    Sophus::SE3d body_T_lidar = lidar_T_body.inverse();
                    int corner_num = 0;
#if 1
                    for (int i = 0; i < laserCloudCornerStackNum; i++)
                    {
                        pointOri = laserCloudCornerStack->points[i];
                        //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
                        pointAssociateToMap(&pointOri, &pointSel);
                        kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                        if (pointSearchSqDis[4] < 1.0)
                        {
                            std::vector<Eigen::Vector3d> nearCorners;
                            Eigen::Vector3d center(0, 0, 0);
                            for (int j = 0; j < 5; j++)
                            {
                                Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
                                        laserCloudCornerFromMap->points[pointSearchInd[j]].y,
                                        laserCloudCornerFromMap->points[pointSearchInd[j]].z);
                                center = center + tmp;
                                nearCorners.push_back(tmp);
                            }
                            center = center / 5.0;

                            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                            for (int j = 0; j < 5; j++)
                            {
                                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                            }

                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                            // if is indeed line feature
                            // note Eigen library sort eigenvalues in increasing order
                            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                            Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                            {
                                Eigen::Vector3d point_on_line = center;
                                Eigen::Vector3d point_a, point_b;
                                point_a = 0.1 * unit_direction + point_on_line;
                                point_b = -0.1 * unit_direction + point_on_line;

                                //ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                                //problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                                auto factor = new LidarLineFactorNeedDistort(curr_point, point_a, point_b, body_T_lidar);
                                problem.AddResidualBlock(factor, loss_function, parameters_j);
                                corner_num++;
                            }
                        }
                    }
#endif

#if 1
                    int surf_num = 0;
                    for (int i = 0; i < laserCloudSurfStackNum; i++)
                    {
                        pointOri = laserCloudSurfStack->points[i];
                        //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
                        pointAssociateToMap(&pointOri, &pointSel);
                        kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                        Eigen::Matrix<double, 5, 3> matA0;
                        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
                        if (pointSearchSqDis[4] < 1.0)
                        {

                            for (int j = 0; j < 5; j++)
                            {
                                matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
                                matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
                                matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
                                //printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
                            }
                            // find the norm of plane
                            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                            double negative_OA_dot_norm = 1 / norm.norm();
                            norm.normalize();

                            // Here n(pa, pb, pc) is unit norm of plane
                            bool planeValid = true;
                            for (int j = 0; j < 5; j++)
                            {
                                // if OX * n > 0.2, then plane is not fit well
                                if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
                                         norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
                                         norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                                {
                                    planeValid = false;
                                    break;
                                }
                            }
                            Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                            if (planeValid)
                            {
                                auto factor = new LidarPlaneFactorNeedDistort(curr_point, norm, body_T_lidar, negative_OA_dot_norm);
                                problem.AddResidualBlock(factor, loss_function, parameters_j);
                                //ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
                                //problem.AddResidualBlock(cost_function, loss_function, parameters_j, parameters_j + 4);
                                surf_num++;
                            }
                        }
                    }
#endif

                    //printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
                    //printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);

                    printf("mapping data assosiation time %f ms \n", t_data.toc());

                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    options.check_gradients = false;
                    options.gradient_check_relative_precision = 1e-4;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);

                    printf("mapping solver time %f ms \n", t_solver.toc());
                    //printf("time %f \n", timeLaserOdometry);
                    //printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
                    //printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
                    //parameters[4], parameters[5], parameters[6]);
                }
                //---Optimize laserMapping do twice

                //+++optimization for IMU
                if(systemInited && lidarState->exist_imu){
                    ceres::Problem::Options problem_options;
                    ceres::Problem problem(problem_options);
                    ceres::LocalParameterization *local_para_se3 = new LocalParameterizationSE3();

                    problem.AddParameterBlock(parameters_j, 7, local_para_se3);
                    std::memcpy(parameters_i , lidarState->wmap_Qbi.data(), sizeof(double) * Sophus::SO3d::num_parameters);
                    std::memcpy(parameters_i + 4, lidarState->wmap_tbi.data(), sizeof(double) * 3);

                    std::memcpy(parameters_speed_bias_i, lidarState->wmap_veci.data(), sizeof(double) * 3);
                    std::memcpy(parameters_speed_bias_i + 3, lidarState->bias_acc_i.data(), sizeof(double) * 3);
                    std::memcpy(parameters_speed_bias_i + 6, lidarState->bias_gyr_i.data(), sizeof(double) * 3);

                    std::memcpy(parameters_speed_bias_j, lidarState->wmap_vecj.data(), sizeof(double) * 3);
                    std::memcpy(parameters_speed_bias_j + 3, lidarState->bias_acc_j.data(), sizeof(double) * 3);
                    std::memcpy(parameters_speed_bias_j + 6, lidarState->bias_gyr_j.data(), sizeof(double) * 3);

                    problem.AddParameterBlock(parameters_i, 7, local_para_se3);
                    problem.AddParameterBlock(parameters_speed_bias_i, 9);
                    problem.AddParameterBlock(parameters_speed_bias_j, 9);
                    problem.SetParameterBlockConstant(parameters_i);
                    problem.SetParameterBlockConstant(parameters_speed_bias_i);
                    problem.SetParameterBlockConstant(parameters_j);

                    //TODO::add IMU preintegration info
                    Eigen::Vector3d gwmap = q_wmap_wodom * gw;
                    auto factor = new IMUFactor(lidarState->imupreinte, gwmap);
                    problem.AddResidualBlock(factor, NULL,
                                             parameters_i,
                                             parameters_speed_bias_i,
                                             parameters_j,
                                             parameters_speed_bias_j);



                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    options.check_gradients = false;
                    options.gradient_check_relative_precision = 1e-4;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);

                    //std::cout << summary.FullReport() <<std::endl;
                    std::cout << "lidarState->bias_acc_j 1= " << lidarState->bias_acc_j << std::endl;
                    std::cout << "lidarState->bias_gyr_j 1= " << lidarState->bias_gyr_j << std::endl;

                    std::memcpy(lidarState->bias_acc_j.data(), parameters_speed_bias_j + 3, sizeof(double) * 3);
                    std::memcpy(lidarState->bias_gyr_j.data(), parameters_speed_bias_j + 6, sizeof(double) * 3);

                    std::cout << "lidarState->bias_acc_j 2= " << lidarState->bias_acc_j << std::endl;
                    std::cout << "lidarState->bias_gyr_j 2= " << lidarState->bias_gyr_j << std::endl;

                }
                //---optimization for IMU

                printf("mapping optimization time %f \n", t_opt.toc());
            }
            else
            {
                ROS_WARN("time Map corner and surf num are not enough");
            }




            //++update pose / baj /bgj ...
            Sophus::SE3d w_T_lidarj = w_T_bodyj * lidar_T_body.inverse();
            q_w_lidarj = w_T_lidarj.so3().unit_quaternion();
            t_w_lidarj = w_T_lidarj.translation();
            w_T_lidarj_last.so3().setQuaternion(q_w_lidarj);
            w_T_lidarj_last.translation() = t_w_lidarj;
            transformUpdate();
            //--update pose / baj /bgj ...

            TicToc t_add;
            for (int i = 0; i < laserCloudCornerStackNum; i++)
            {
                pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

                int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
                int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
                int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

                if (pointSel.x + 25.0 < 0)
                    cubeI--;
                if (pointSel.y + 25.0 < 0)
                    cubeJ--;
                if (pointSel.z + 25.0 < 0)
                    cubeK--;

                if (cubeI >= 0 && cubeI < laserCloudWidth &&
                        cubeJ >= 0 && cubeJ < laserCloudHeight &&
                        cubeK >= 0 && cubeK < laserCloudDepth)
                {
                    int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                    laserCloudCornerArray[cubeInd]->push_back(pointSel);
                }
            }

            for (int i = 0; i < laserCloudSurfStackNum; i++)
            {
                pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

                int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
                int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
                int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

                if (pointSel.x + 25.0 < 0)
                    cubeI--;
                if (pointSel.y + 25.0 < 0)
                    cubeJ--;
                if (pointSel.z + 25.0 < 0)
                    cubeK--;

                if (cubeI >= 0 && cubeI < laserCloudWidth &&
                        cubeJ >= 0 && cubeJ < laserCloudHeight &&
                        cubeK >= 0 && cubeK < laserCloudDepth)
                {
                    int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                    laserCloudSurfArray[cubeInd]->push_back(pointSel);
                }
            }
            printf("add points time %f ms\n", t_add.toc());


            TicToc t_filter;
            for (int i = 0; i < laserCloudValidNum; i++)
            {
                int ind = laserCloudValidInd[i];

                pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
                downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
                downSizeFilterCorner.filter(*tmpCorner);
                laserCloudCornerArray[ind] = tmpCorner;

                pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
                downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
                downSizeFilterSurf.filter(*tmpSurf);
                laserCloudSurfArray[ind] = tmpSurf;
            }
            printf("filter time %f ms \n", t_filter.toc());

            TicToc t_pub;
            //publish surround map for every 5 frame
            if (frameCount % 5 == 0)
            {
                laserCloudSurround->clear();
                for (int i = 0; i < laserCloudSurroundNum; i++)
                {
                    int ind = laserCloudSurroundInd[i];
                    *laserCloudSurround += *laserCloudCornerArray[ind];
                    *laserCloudSurround += *laserCloudSurfArray[ind];
                }

                sensor_msgs::PointCloud2 laserCloudSurround3;
                pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
                laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                laserCloudSurround3.header.frame_id = "/world";
                pubLaserCloudSurround.publish(laserCloudSurround3);
            }

            if (frameCount % 20 == 0)
            {
                pcl::PointCloud<PointType> laserCloudMap;
                for (int i = 0; i < 4851; i++)
                {
                    laserCloudMap += *laserCloudCornerArray[i];
                    laserCloudMap += *laserCloudSurfArray[i];
                }
                sensor_msgs::PointCloud2 laserCloudMsg;
                pcl::toROSMsg(laserCloudMap, laserCloudMsg);
                laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
                laserCloudMsg.header.frame_id = "/world";
                pubLaserCloudMap.publish(laserCloudMsg);
            }

#if !PUB_ORI_PCL2
            sensor_msgs::PointCloud2 laserCloudFullRes_voxblox;
            pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes_voxblox);
            laserCloudFullRes_voxblox.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            laserCloudFullRes_voxblox.header.frame_id = "/velodyne";
            pubLaserCloudvoxblox.publish(laserCloudFullRes_voxblox);
#endif


            int laserCloudFullResNum = laserCloudFullRes->points.size();
            for (int i = 0; i < laserCloudFullResNum; i++)
            {
                pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
            }

            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
            laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            laserCloudFullRes3.header.frame_id = "/world";
            pubLaserCloudFullRes.publish(laserCloudFullRes3);

            printf("mapping pub time %f ms \n", t_pub.toc());

            printf("whole mapping time %f ms +++++\n", t_whole.toc());

            nav_msgs::Odometry odomAftMapped;
            odomAftMapped.header.frame_id = "/world";
            odomAftMapped.child_frame_id = "/aft_mapped";
            odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            odomAftMapped.pose.pose.orientation.x = q_w_lidarj.x();
            odomAftMapped.pose.pose.orientation.y = q_w_lidarj.y();
            odomAftMapped.pose.pose.orientation.z = q_w_lidarj.z();
            odomAftMapped.pose.pose.orientation.w = q_w_lidarj.w();
            odomAftMapped.pose.pose.position.x = t_w_lidarj.x();
            odomAftMapped.pose.pose.position.y = t_w_lidarj.y();
            odomAftMapped.pose.pose.position.z = t_w_lidarj.z();
            pubOdomAftMapped.publish(odomAftMapped);

            geometry_msgs::TransformStamped transform_stamped;
            transform_stamped.header = odomAftMapped.header;
            transform_stamped.transform.rotation.w = q_w_lidarj.w();
            transform_stamped.transform.rotation.x = q_w_lidarj.x();
            transform_stamped.transform.rotation.y = q_w_lidarj.y();
            transform_stamped.transform.rotation.z = q_w_lidarj.z();
            transform_stamped.transform.translation.x = t_w_lidarj.x();
            transform_stamped.transform.translation.y = t_w_lidarj.y();
            transform_stamped.transform.translation.z = t_w_lidarj.z();
            pub_TF_stamped.publish(transform_stamped);


            geometry_msgs::PoseStamped laserAfterMappedPose;
            laserAfterMappedPose.header = odomAftMapped.header;
            laserAfterMappedPose.pose = odomAftMapped.pose.pose;
            laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
            laserAfterMappedPath.header.frame_id = "/world";
            laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
            pubLaserAfterMappedPath.publish(laserAfterMappedPath);

            static tf::TransformBroadcaster br;
            tf::Transform transform;
            tf::Quaternion q;
            transform.setOrigin(tf::Vector3(t_w_lidarj(0),
                                            t_w_lidarj(1),
                                            t_w_lidarj(2)));
            q.setW(q_w_lidarj.w());
            q.setX(q_w_lidarj.x());
            q.setY(q_w_lidarj.y());
            q.setZ(q_w_lidarj.z());
            transform.setRotation(q);
            br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/world", "/aft_mapped"));

            frameCount++;
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    float lineRes = 0;
    float planeRes = 0;
    nh.param<float>("mapping_line_resolution", lineRes, 0.4);
    nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
    printf("line resolution %f plane resolution %f \n", lineRes, planeRes);
    downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
    downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);
    GetParam("IMU_INFO/acc_n", acc_n);
    GetParam("IMU_INFO/gyr_n", gyr_n);
    GetParam("IMU_INFO/acc_w", acc_w);
    GetParam("IMU_INFO/gyr_w", gyr_w);
    GetParam("IMU_INFO/g_norm", g_norm);
    gw.z() = g_norm;
    gw.x() = 0;
    gw.y() = 0;

    double qw, qx, qy, qz, tx, ty, tz;
    GetParam("/lidar_T_body/tx", tx);
    GetParam("/lidar_T_body/ty", ty);
    GetParam("/lidar_T_body/tz", tz);
    GetParam("/lidar_T_body/qx", qx);
    GetParam("/lidar_T_body/qy", qy);
    GetParam("/lidar_T_body/qz", qz);
    GetParam("/lidar_T_body/qw", qw);
    lidar_T_body.so3().setQuaternion(Eigen::Quaterniond(qw, qx, qy, qz));
    lidar_T_body.translation().x() = tx;
    lidar_T_body.translation().y() = ty;
    lidar_T_body.translation().z() = tz;

    //ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);

    ros::Subscriber subPoseijIMU = nh.subscribe<add_msg::RelativePoseIMU>("/Poseij_imu", 100, relativePoseIMUHandler);

    ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud[2] {{nh, "/laser_cloud_corner_last", 100},
                                                                        {nh, "/velodyne_points", 100}};
    message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> sync(sub_cloud[0], sub_cloud[1], 100);
    sync.registerCallback(boost::bind(&laserCloudCornerLastHandler, _1, _2));


    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);

    pub_TF_stamped = nh.advertise<geometry_msgs::TransformStamped>("GT_transform", 1000);

    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);

    pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

    pubLaserCloudvoxblox = nh.advertise<sensor_msgs::PointCloud2>("/voxblox_points", 100);

    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);

    pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);

    pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

    for (int i = 0; i < laserCloudNum; i++)
    {
        laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
        laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
    }

    std::thread mapping_process{process};

    ros::spin();

    return 0;
}
