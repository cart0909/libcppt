#include "ros/ros.h"
#include "global_opt.h"
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <iostream>
#include <stdio.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>
#include <queue>
#include <mutex>
#include "global_opt.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

GlobalOptimizePtr globalOptimze;
std::queue<sensor_msgs::NavSatFixConstPtr> gpsQueue;
std::mutex m_buf;
Eigen::Vector3d last_t(-100, -100, -100);
double last_vio_t = -1;
ros::Publisher pub_global_path, pub_global_odometry, only_gps_odometry, pub_best_fusion;
nav_msgs::Path *global_path;

//Initalize distance for gps
float SKIP_DIS = 0.5;
void Sync_callback(const nav_msgs::OdometryConstPtr &pose_odom, const sensor_msgs::NavSatFixConstPtr &gps_msg){

    double t = pose_odom->header.stamp.toSec();
    Eigen::Vector3d vio_t(pose_odom->pose.pose.position.x, pose_odom->pose.pose.position.y, pose_odom->pose.pose.position.z);
    Eigen::Quaterniond vio_q;
    vio_q.w() = pose_odom->pose.pose.orientation.w;
    vio_q.x() = pose_odom->pose.pose.orientation.x;
    vio_q.y() = pose_odom->pose.pose.orientation.y;
    vio_q.z() = pose_odom->pose.pose.orientation.z;
    //Current pose of wvio_T_body
    Sophus::SE3d wvio_T_b(vio_q, vio_t);

    if((vio_t - last_t).norm() > SKIP_DIS){
        //into optimize process
        globalOptimze->inputOdom(t, wvio_T_b);
        double latitude = gps_msg->latitude;
        double longitude = gps_msg->longitude;
        double altitude = gps_msg->altitude;
        double pos_accuracy = gps_msg->position_covariance[0];
        if(pos_accuracy <= 0)
            pos_accuracy = 1;
        globalOptimze->inputGPS(t, latitude, longitude, altitude, pos_accuracy);
        last_t = vio_t;
        //Get GPSXYZ
        std::vector<double> gps_xyz;
        globalOptimze->getGPSXYZ(t, gps_xyz);
        //Pub global path to rviz.
#if 0
        Sophus::SE3d wgps_T_curbody;
        globalOptimze->getGlobalOdom(wgps_T_curbody);
        nav_msgs::Odometry odometry;
        odometry.header = pose_odom->header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        odometry.pose.pose.position.x = wgps_T_curbody.translation().x();
        odometry.pose.pose.position.y = wgps_T_curbody.translation().y();
        odometry.pose.pose.position.z = wgps_T_curbody.translation().z();
        odometry.pose.pose.orientation.x = wgps_T_curbody.so3().unit_quaternion().x();
        odometry.pose.pose.orientation.y = wgps_T_curbody.so3().unit_quaternion().y();
        odometry.pose.pose.orientation.z = wgps_T_curbody.so3().unit_quaternion().z();
        odometry.pose.pose.orientation.w = wgps_T_curbody.so3().unit_quaternion().w();
        pub_global_odometry.publish(odometry);
#else
        nav_msgs::Odometry odometry;
        odometry.header = pose_odom->header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        odometry.pose.pose.position.x = gps_xyz[0];
        odometry.pose.pose.position.y = gps_xyz[1];
        odometry.pose.pose.position.z = gps_xyz[2];
        odometry.pose.pose.orientation.x = 0;
        odometry.pose.pose.orientation.y = 0;
        odometry.pose.pose.orientation.z = 0;
        odometry.pose.pose.orientation.w = 1;
        only_gps_odometry.publish(odometry);
#endif
        pub_global_path.publish(*global_path);
    }
    else{
        Sophus::SE3d wgps_T_wvio;
        globalOptimze->getWGPS_T_WVIO(wgps_T_wvio);
        Sophus::SE3d wgps_T_curbody;
        wgps_T_curbody = wgps_T_wvio * wvio_T_b;
        globalOptimze->AddPoseGlobalPath(t, wgps_T_curbody);
        nav_msgs::Odometry odometry;
        odometry.header = pose_odom->header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        odometry.pose.pose.position.x = wgps_T_curbody.translation().x();
        odometry.pose.pose.position.y = wgps_T_curbody.translation().y();
        odometry.pose.pose.position.z = wgps_T_curbody.translation().z();
        odometry.pose.pose.orientation.x = wgps_T_curbody.so3().unit_quaternion().x();
        odometry.pose.pose.orientation.y = wgps_T_curbody.so3().unit_quaternion().y();
        odometry.pose.pose.orientation.z = wgps_T_curbody.so3().unit_quaternion().z();
        odometry.pose.pose.orientation.w = wgps_T_curbody.so3().unit_quaternion().w();
        pub_global_odometry.publish(odometry);
        pub_global_path.publish(*global_path);
    }
}


void GPS_callback(const sensor_msgs::NavSatFixConstPtr &GPS_msg)
{
    //printf("! \n");
    m_buf.lock();
    gpsQueue.push(GPS_msg);
    m_buf.unlock();
}



int main(int argc, char** argv){
    ros::init(argc, argv, "GlobalEstimator");
    ros::NodeHandle nh("~");
    globalOptimze = std::make_shared<GlobalOptimize>();
    pub_global_path = nh.advertise<nav_msgs::Path>("global_path", 100);
    pub_global_odometry = nh.advertise<nav_msgs::Odometry>("global_odometry", 10);
    only_gps_odometry = nh.advertise<nav_msgs::Odometry>("only_gps_xyz", 10);
    pub_best_fusion = nh.advertise<nav_msgs::Odometry>("best_fusion_withGps", 10);
    global_path = &globalOptimze->global_path;

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/cppt_reloc_wb", 300);
    message_filters::Subscriber<sensor_msgs::NavSatFix> gps_sub(nh, "/fix", 100);
    //message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/laser_odom_to_init", 300);
    //message_filters::Subscriber<sensor_msgs::NavSatFix> gps_sub(nh, "/kitti/oxts/gps/fix", 100);

    typedef  message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::NavSatFix> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), odom_sub, gps_sub);
    sync.registerCallback(boost::bind(&Sync_callback, _1, _2));
    ros::spin();
    return  0;
}
