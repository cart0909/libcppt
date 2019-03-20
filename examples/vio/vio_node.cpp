#include <fstream>
#include <queue>
#include <thread>
#include <condition_variable>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h>
// catch ctrl+c signal
#include <signal.h>

#include "ros_utility.h"
#include "src/CameraPoseVisualization.h"
#include "src/visualization.h"
#include "pl_system.h"
using namespace std;
using namespace message_filters;
using namespace sensor_msgs;

// global variable
PLSystemPtr pl_system;

void ImageCallback(const ImageConstPtr& img_msg, const ImageConstPtr& img_r_msg) {
//    if(image_timestamp != -1.0f && std::abs(img_msg->header.stamp.toSec() - image_timestamp) > 10) { // 10 ms
//        // reset system
//        ROS_WARN_STREAM("System detect different image flow, trying to reset the system.");
//        system->Reset();
//    }
    cv::Mat img_l, img_r;
    img_l = cv_bridge::toCvCopy(img_msg, "mono8")->image;
    img_r = cv_bridge::toCvCopy(img_r_msg, "mono8")->image;
//        img_l = cv_bridge::toCvCopy(img_msg, "8UC1")->image;
//        img_r = cv_bridge::toCvCopy(img_r_msg, "8UC1")->image;
    pl_system->PushImages(img_l, img_r, img_msg->header.stamp.toSec());
}

void ImuCallback(const ImuConstPtr& imu_msg) {
    Eigen::Vector3d gyr = Eigen::Vector3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    Eigen::Vector3d acc = Eigen::Vector3d(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    pl_system->PushImuData(gyr, acc, imu_msg->header.stamp.toSec());
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "cppt_vio", ros::init_options::NoSigintHandler);
//    ros::init(argc, argv, "cppt_vio");
//    signal(SIGINT, sigint_handler);
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    std::string config_file;
    config_file = readParam<std::string>(nh, "config_file");
    pl_system = std::make_shared<PLSystem>(config_file);

    message_filters::Subscriber<Image> sub_img[2] {{nh, "/camera/left/image_raw", 100},
                                                   {nh, "/camera/right/image_raw", 100}};
    TimeSynchronizer<Image, Image> sync(sub_img[0], sub_img[1], 100);
    sync.registerCallback(boost::bind(&ImageCallback, _1, _2));

    ros::Subscriber sub_imu = nh.subscribe("/imu/data_raw", 2000, &ImuCallback,
                                           ros::TransportHints().tcpNoDelay());
    vis::ReadFromNodeHandle(nh, pl_system);
    ROS_INFO_STREAM("Player is ready.");
    ros::spin();
    return 0;
}

//    node.pub_keyframes = nh.advertise<visualization_msgs::Marker>("keyframes", 1000);

//    void PubSlidingWindow(uint64_t seq, double timestamp, const std::vector<Sophus::SE3d>& v_Twc) {
//        if(v_Twc.empty())
//            return;

//        { // print keyframe point
//            visualization_msgs::Marker key_poses;
//            key_poses.header.frame_id = "world";
//            key_poses.header.seq = seq;
//            key_poses.header.stamp.fromSec(timestamp);
//            key_poses.ns = "keyframes";
//            key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
//            key_poses.action = visualization_msgs::Marker::ADD;
//            key_poses.pose.orientation.w = 1.0;
//            key_poses.lifetime = ros::Duration();

//            key_poses.id = 0;
//            key_poses.scale.x = 0.05;
//            key_poses.scale.y = 0.05;
//            key_poses.scale.z = 0.05;
//            key_poses.color.r = 1.0;
//            key_poses.color.a = 1.0;

//            for(auto& Twc : v_Twc) {
//                Eigen::Vector3d twc = Twc.translation();
//                geometry_msgs::Point pose_marker;
//                pose_marker.x = twc(0);
//                pose_marker.y = twc(1);
//                pose_marker.z = twc(2);
//                key_poses.points.emplace_back(pose_marker);
//            }
//            pub_keyframes.publish(key_poses);
//        }
//    }

//void sigint_handler(int s) {
//    ROS_INFO_STREAM("logging trajectory to the file");
//    std::ofstream fout(node.log_filename);
//    if(!fout.is_open())
//        exit(1);

//    for(auto& pose : node.path.poses) {
//        fout << pose.header.stamp << " ";
//        fout << pose.pose.position.x << " " << pose.pose.position.y << " " <<
//                pose.pose.position.z << " ";
//        fout << pose.pose.orientation.x << " " << pose.pose.orientation.y << " "
//             << pose.pose.orientation.z << " " << pose.pose.orientation.w << std::endl;
//    }

//    exit(1);
//}
