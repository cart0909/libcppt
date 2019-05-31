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
#define SKIP_DIS 0.02

GlobalOptimizePtr globalOptimze;
std::queue<sensor_msgs::NavSatFixConstPtr> gpsQueue;
std::mutex m_buf;
Eigen::Vector3d last_t(-100, -100, -100);
double last_vio_t = -1;
ros::Publisher pub_global_path, pub_global_odometry;
nav_msgs::Path *global_path;
void Sync_callback(const nav_msgs::OdometryConstPtr &pose_odom, const sensor_msgs::NavSatFixConstPtr &gps_msg){

    double t = pose_odom->header.stamp.toSec();
    double gps_t = gps_msg->header.stamp.toSec();
    if(gps_t >= t - 0.01 && gps_t <= t + 0.01){
        std::cout << "cannot find sync, check sensor timestamp" <<std::endl;
        return;
    }

    Eigen::Vector3d vio_t(pose_odom->pose.pose.position.x, pose_odom->pose.pose.position.y, pose_odom->pose.pose.position.z);
    Eigen::Quaterniond vio_q;
    vio_q.w() = pose_odom->pose.pose.orientation.w;
    vio_q.x() = pose_odom->pose.pose.orientation.x;
    vio_q.y() = pose_odom->pose.pose.orientation.y;
    vio_q.z() = pose_odom->pose.pose.orientation.z;

    if((vio_t - last_t).norm() > SKIP_DIS){
        Sophus::SE3d wvio_T_b(vio_q, vio_t);
        globalOptimze->inputOdom(t, wvio_T_b);
        double latitude = gps_msg->latitude;
        double longitude = gps_msg->longitude;
        double altitude = gps_msg->altitude;
        double pos_accuracy = gps_msg->position_covariance[0];
        if(pos_accuracy <= 0)
            pos_accuracy = 1;
        globalOptimze->inputGPS(t, latitude, longitude, altitude, pos_accuracy);
        last_t = vio_t;

        //Pub global path to rviz.
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

void VIO_callback(const nav_msgs::OdometryConstPtr &pose_odom){
    double t = pose_odom->header.stamp.toSec();
    last_vio_t = t;
    Eigen::Vector3d vio_t(pose_odom->pose.pose.position.x, pose_odom->pose.pose.position.y, pose_odom->pose.pose.position.z);
    Eigen::Quaterniond vio_q;
    vio_q.w() = pose_odom->pose.pose.orientation.w;
    vio_q.x() = pose_odom->pose.pose.orientation.x;
    vio_q.y() = pose_odom->pose.pose.orientation.y;
    vio_q.z() = pose_odom->pose.pose.orientation.z;
    Sophus::SE3d wvio_T_b(vio_q, vio_t);
    //std::cout << "wvio_T_b=" << wvio_T_b.matrix() <<std::endl;
    globalOptimze->inputOdom(t, wvio_T_b);

    //find the matching of gps info.
    m_buf.lock();
    while(!gpsQueue.empty())
    {
        sensor_msgs::NavSatFixConstPtr GPS_msg = gpsQueue.front();
        double gps_t = GPS_msg->header.stamp.toSec();

        if(gps_t >= t - 0.1 && gps_t <= t + 0.1)
        {
            double latitude = GPS_msg->latitude;
            double longitude = GPS_msg->longitude;
            double altitude = GPS_msg->altitude;

            double pos_accuracy = GPS_msg->position_covariance[0];
            if(pos_accuracy <= 0)
                pos_accuracy = 1;

            globalOptimze->inputGPS(t, latitude, longitude, altitude, pos_accuracy);
            gpsQueue.pop();
            break;
        }
        else if(gps_t < t - 0.01) //when gps_info too old, remove that.
            gpsQueue.pop();
        else if(gps_t > t + 0.01)
            break;
    }
    m_buf.unlock();
}





int main(int argc, char** argv){
    ros::init(argc, argv, "GlobalEstimator");
    ros::NodeHandle nh("~");
    //Node GlobalEstimator(nh);
    globalOptimze = std::make_shared<GlobalOptimize>();
    pub_global_path = nh.advertise<nav_msgs::Path>("global_path", 100);
    pub_global_odometry = nh.advertise<nav_msgs::Odometry>("global_odometry", 100);
    global_path = &globalOptimze->global_path;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/cppt_reloc_wb", 100);
    message_filters::Subscriber<sensor_msgs::NavSatFix> gps_sub(nh, "/fix", 100);

    typedef  message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::NavSatFix> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), odom_sub, gps_sub);
    sync.registerCallback(boost::bind(&Sync_callback, _1, _2));
    ros::spin();
    return  0;
}
