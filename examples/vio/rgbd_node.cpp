#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "ros_utility.h"
#include "rgbd_system.h"
#include "visualization.h"
using namespace std;
using namespace ros;
using namespace message_filters;
using namespace sensor_msgs;

RGBDSystemPtr system_ptr;

void ImageCallback(const ImageConstPtr& rgb_msg, const ImageConstPtr& depth_msg) {
    cv::Mat rgb_img = cv_bridge::toCvShare(rgb_msg, "bgr8")->image;
    cv::Mat gray_img;
    cv::cvtColor(rgb_img, gray_img, CV_BGR2GRAY);
    cv::Mat depth_img = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    system_ptr->PushImages(gray_img, depth_img, rgb_msg->header.stamp.toSec());
}

void ImuCallback(const ImuConstPtr& imu_msg) {
    Eigen::Vector3d gyr(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z),
                    acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    system_ptr->PushImuData(gyr, acc, imu_msg->header.stamp.toSec());
}

int main(int argc, char** argv) {
    init(argc, argv, "rgbd_node", init_options::NoSigintHandler);
    NodeHandle nh("~");
    console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, console::levels::Info);

    message_filters::Subscriber<Image> sub_rgb_img(nh, "/camera/rgb/image_raw", 100),
                                       sub_depth_img(nh, "/camera/depth/image_raw", 100);
    TimeSynchronizer<Image, Image> time_sync(sub_rgb_img, sub_depth_img, 100);
    time_sync.registerCallback(&ImageCallback);

    ros::Subscriber sub_imu = nh.subscribe("/imu/data_raw", 2000, &ImuCallback,
                                           TransportHints().tcpNoDelay());

    std::string config_file = readParam<std::string>(nh, "config_file");;
    system_ptr = std::make_shared<RGBDSystem>(config_file);
    vis::ReadFromNodeHandle(nh, system_ptr);
    ROS_INFO_STREAM("RGB-D Player is ready");
    spin();
    return 0;
}
