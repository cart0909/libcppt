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
// catch ctrl+c signal
#include <signal.h>

#include "ros_utility.h"
#include "src/CameraPoseVisualization.h"
#include "system.h"
using namespace std;
using namespace message_filters;
using namespace sensor_msgs;

class Node {
public:
    using Measurements = vector<pair<pair<ImageConstPtr, ImageConstPtr>, vector<ImuConstPtr>>>;
    Node() : vio_pose_visual(0, 1, 0, 1), reloc_pose_visual(1, 0, 0, 1), loop_edge_visual(0, 1, 0, 1),
        fast_pose_visual(1, 1, 0, 1)
    {
        vio_pose_visual.setScale(0.3);
        fast_pose_visual.setScale(0.3);
    }
    ~Node() {}

    void ReadFromNodeHandle(ros::NodeHandle& nh) {
        std::string config_file;
        config_file = readParam<std::string>(nh, "config_file");

        cv::FileStorage fs(config_file, cv::FileStorage::READ);
        fs["imu_topic"] >> imu_topic;
        fs["image_topic"] >> img_topic[0];
        fs["image_r_topic"] >> img_topic[1];
        fs["output_path"] >> log_filename;
        fs.release();

        system = std::make_shared<System>(config_file);
        system->SubTrackingImg(std::bind(&Node::PubTrackImg, this, std::placeholders::_1,
                                         std::placeholders::_2));
        system->SubVIOTwc(std::bind(&Node::PubCurPose, this, std::placeholders::_1,
                                    std::placeholders::_2));
        system->SubRelocTwc(std::bind(&Node::PubRelocPose, this, std::placeholders::_1,
                                      std::placeholders::_2));
        system->SubAddRelocPath(std::bind(&Node::AddRelocPath, this, std::placeholders::_1));
        system->SubUpdateRelocPath(std::bind(&Node::UpdateRelocPath, this, std::placeholders::_1));
        system->SubLoopEdge(std::bind(&Node::PushLoopEdgeIndex, this, std::placeholders::_1));
        system->SubRelocImg(std::bind(&Node::PubRelocImg, this, std::placeholders::_1));
    }

    void ImageCallback(const ImageConstPtr& img_msg, const ImageConstPtr& img_r_msg) {
        if(image_timestamp != -1.0f && std::abs(img_msg->header.stamp.toSec() - image_timestamp) > 10) { // 10 ms
            // reset system
            ROS_WARN_STREAM("System detect different image flow, trying to reset the system.");
            system->Reset();
        }
        cv::Mat img_l, img_r;
        img_l = cv_bridge::toCvCopy(img_msg, "mono8")->image;
        img_r = cv_bridge::toCvCopy(img_r_msg, "mono8")->image;
        system->PushImages(img_l, img_r, img_msg->header.stamp.toSec());
    }

    void ImuCallback(const ImuConstPtr& imu_msg) {
        Eigen::Vector3d gyr = Eigen::Vector3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
        Eigen::Vector3d acc = Eigen::Vector3d(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
        system->PushImuData(gyr, acc, imu_msg->header.stamp.toSec());

        Sophus::SE3d Twc;
        if(system->Predict(Eigen::Vector3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z),
                           Eigen::Vector3d(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z),
                           imu_msg->header.stamp.toSec(), Twc))
        {
            PubFastPose(imu_msg->header.stamp.toSec(), Twc);
        }
    }

    void PubTrackImg(double timestamp, const cv::Mat& track_img) {
        cv_bridge::CvImage track_cvimage;
        track_cvimage.header.frame_id = "world";
        track_cvimage.header.stamp.fromSec(timestamp);
        track_cvimage.image = track_img;
        track_cvimage.encoding = sensor_msgs::image_encodings::BGR8;
        pub_track_img.publish(track_cvimage.toImageMsg());
    }

    void PubRelocImg(const cv::Mat& reloc_img) {
        cv_bridge::CvImage reloc_cvimg;
        reloc_cvimg.header.frame_id = "world";
        reloc_cvimg.image = reloc_img;
        reloc_cvimg.encoding = sensor_msgs::image_encodings::BGR8;
        pub_reloc_img.publish(reloc_cvimg.toImageMsg());
    }

    void PubCurPose(double timestamp, const Sophus::SE3d& Twc) {
        // public latest frame
        Eigen::Vector3d twc = Twc.translation();
        Eigen::Quaterniond qwc = Twc.so3().unit_quaternion();
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = "world";
        pose_stamped.header.stamp.fromSec(timestamp);
        pose_stamped.pose.orientation.w = qwc.w();
        pose_stamped.pose.orientation.x = qwc.x();
        pose_stamped.pose.orientation.y = qwc.y();
        pose_stamped.pose.orientation.z = qwc.z();
        pose_stamped.pose.position.x = twc(0);
        pose_stamped.pose.position.y = twc(1);
        pose_stamped.pose.position.z = twc(2);
        vio_path.poses.emplace_back(pose_stamped);
        pub_vio_path.publish(vio_path);

        // camera pose
        vio_pose_visual.reset();
        vio_pose_visual.add_pose(twc, qwc);
        vio_pose_visual.publish_by(pub_vio_pose, vio_path.header);
    }

    void PubFastPose(double timestamp, const Sophus::SE3d& Twc) {
        std_msgs::Header header;
        header.frame_id = "world";
        header.stamp.fromSec(timestamp);
        Eigen::Vector3d twc = Twc.translation();
        Eigen::Quaterniond qwc = Twc.unit_quaternion();
        fast_pose_visual.reset();
        fast_pose_visual.add_pose(twc, qwc);
        fast_pose_visual.publish_by(pub_fast_pose, header);

        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped.header = header;
        transform_stamped.transform.rotation.w = qwc.w();
        transform_stamped.transform.rotation.x = qwc.x();
        transform_stamped.transform.rotation.y = qwc.y();
        transform_stamped.transform.rotation.z = qwc.z();

        transform_stamped.transform.translation.x = twc(0);
        transform_stamped.transform.translation.y = twc(1);
        transform_stamped.transform.translation.z = twc(2);
        pub_transform.publish(transform_stamped);
    }

    void PubRelocPose(double timestamp, const Sophus::SE3d& Twc) {
        std_msgs::Header header;
        header.frame_id = "world";
        header.stamp.fromSec(timestamp);
        Eigen::Vector3d twc = Twc.translation();
        Eigen::Quaterniond qwc = Twc.unit_quaternion();
        reloc_pose_visual.reset();
        reloc_pose_visual.add_pose(twc, qwc);
        reloc_pose_visual.publish_by(pub_reloc_pose, header);
    }

    void AddRelocPath(const Sophus::SE3d& Twc) {
        Eigen::Vector3d twc = Twc.translation();
        Eigen::Quaterniond qwc = Twc.unit_quaternion();
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose.orientation.w = qwc.w();
        pose_stamped.pose.orientation.x = qwc.x();
        pose_stamped.pose.orientation.y = qwc.y();
        pose_stamped.pose.orientation.z = qwc.z();
        pose_stamped.pose.position.x = twc(0);
        pose_stamped.pose.position.y = twc(1);
        pose_stamped.pose.position.z = twc(2);
        reloc_path.poses.emplace_back(pose_stamped);
        pub_reloc_path.publish(reloc_path);

        loop_edge_visual.reset();
        mtx_loop_edge_index.lock();
        for(auto& it : v_loop_edge_index) {
            auto& pos_i = reloc_path.poses[it.first].pose.position,
                  pos_j = reloc_path.poses[it.second].pose.position;
            loop_edge_visual.add_loopedge(Eigen::Vector3d(pos_i.x, pos_i.y, pos_i.z),
                                          Eigen::Vector3d(pos_j.x, pos_j.y, pos_j.z));
        }
        mtx_loop_edge_index.unlock();
        loop_edge_visual.publish_by(pub_loop_edge, pose_stamped.header);
    }

    void UpdateRelocPath(const std::vector<Sophus::SE3d>& v_Twc) {
        std_msgs::Header header;
        header.frame_id = "world";
        reloc_path.poses.clear();
        for(auto& Twc : v_Twc) {
            Eigen::Vector3d twc = Twc.translation();
            Eigen::Quaterniond qwc = Twc.unit_quaternion();
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.frame_id = "world";
            pose_stamped.pose.orientation.w = qwc.w();
            pose_stamped.pose.orientation.x = qwc.x();
            pose_stamped.pose.orientation.y = qwc.y();
            pose_stamped.pose.orientation.z = qwc.z();
            pose_stamped.pose.position.x = twc(0);
            pose_stamped.pose.position.y = twc(1);
            pose_stamped.pose.position.z = twc(2);
            reloc_path.poses.emplace_back(pose_stamped);
        }
        pub_reloc_path.publish(reloc_path);

        loop_edge_visual.reset();
        mtx_loop_edge_index.lock();
        for(auto& it : v_loop_edge_index) {
            auto& pos_i = reloc_path.poses[it.first].pose.position,
                  pos_j = reloc_path.poses[it.second].pose.position;
            loop_edge_visual.add_loopedge(Eigen::Vector3d(pos_i.x, pos_i.y, pos_i.z),
                                          Eigen::Vector3d(pos_j.x, pos_j.y, pos_j.z));
        }
        mtx_loop_edge_index.unlock();
        loop_edge_visual.publish_by(pub_loop_edge, header);
    }

    void PushLoopEdgeIndex(const std::pair<uint64_t, uint64_t>& edge) {
        unique_lock<mutex> lock(mtx_loop_edge_index);
        v_loop_edge_index.emplace_back(edge);
    }

    string imu_topic;
    string img_topic[2];
    string log_filename;

    SystemPtr system;

    ros::Publisher pub_track_img;
    ros::Publisher pub_vio_path;
    nav_msgs::Path vio_path;
    ros::Publisher pub_vio_pose;
    CameraPoseVisualization vio_pose_visual;

    double image_timestamp = -1.0f;

    // show reloc
    nav_msgs::Path reloc_path;
    CameraPoseVisualization reloc_pose_visual;
    CameraPoseVisualization loop_edge_visual;
    ros::Publisher pub_reloc_path;
    ros::Publisher pub_reloc_pose;
    ros::Publisher pub_loop_edge;
    ros::Publisher pub_reloc_img;

    mutex mtx_loop_edge_index;
    vector<pair<uint64_t, uint64_t>> v_loop_edge_index;

    // fast pose
    CameraPoseVisualization fast_pose_visual;
    ros::Publisher pub_fast_pose;

    // for dense mapping
    ros::Publisher pub_transform;
};

// global variable
Node node;

int main(int argc, char** argv) {
    ros::init(argc, argv, "cppt_vio", ros::init_options::NoSigintHandler);
//    ros::init(argc, argv, "cppt_vio");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

//    signal(SIGINT, sigint_handler);

    node.ReadFromNodeHandle(nh);

    message_filters::Subscriber<Image> sub_img[2] {{nh, node.img_topic[0], 100},
                                                   {nh, node.img_topic[1], 100}};
    TimeSynchronizer<Image, Image> sync(sub_img[0], sub_img[1], 100);
    sync.registerCallback(boost::bind(&Node::ImageCallback, &node, _1, _2));

    ros::Subscriber sub_imu = nh.subscribe(node.imu_topic, 2000, &Node::ImuCallback, &node,
                                           ros::TransportHints().tcpNoDelay());

    node.pub_track_img = nh.advertise<sensor_msgs::Image>("track_img", 1000);
    node.pub_vio_path = nh.advertise<nav_msgs::Path>("vio_path", 1000);
    node.vio_path.header.frame_id = "world";
    node.pub_vio_pose = nh.advertise<visualization_msgs::MarkerArray>("vio_pose", 1000);
    node.pub_reloc_path = nh.advertise<nav_msgs::Path>("reloc_path", 1000);
    node.reloc_path.header.frame_id = "world";
    node.pub_reloc_pose = nh.advertise<visualization_msgs::MarkerArray>("reloc_pose", 1000);
    node.pub_loop_edge = nh.advertise<visualization_msgs::MarkerArray>("loop_edge", 1000);
    node.pub_reloc_img = nh.advertise<sensor_msgs::Image>("reloc_img", 1000);
    node.pub_fast_pose = nh.advertise<visualization_msgs::MarkerArray>("fast_pose", 1000);
    node.pub_transform = nh.advertise<geometry_msgs::TransformStamped>("transform", 1000);
    ROS_INFO_STREAM("Player is ready.");

    ros::spin();
    return 0;
}







//    node.pub_keyframes = nh.advertise<visualization_msgs::Marker>("keyframes", 1000);
//    node.pub_mappoints = nh.advertise<sensor_msgs::PointCloud>("mappoints", 1000);
//    node.pub_margin_mps = nh.advertise<sensor_msgs::PointCloud>("margin_mps", 1000);

// =======================================================================================
//    void PubMapPoint(uint64_t seq, double timestamp, const Eigen::VecVector3d& mps) {
//        sensor_msgs::PointCloud mps_msg;
//        mps_msg.header.frame_id = "world";
//        mps_msg.header.seq = seq;
//        mps_msg.header.stamp.fromSec(timestamp);

//        for(auto& x3Dw : mps) {
//            geometry_msgs::Point32 point_marker;
//            point_marker.x = x3Dw(0);
//            point_marker.y = x3Dw(1);
//            point_marker.z = x3Dw(2);
//            mps_msg.points.emplace_back(point_marker);
//        }
//        pub_mappoints.publish(mps_msg);
//    }

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

//    void PubMarginMps(uint64_t seq, double timestamp, const Eigen::VecVector3d& margin_mps) {
//        sensor_msgs::PointCloud margin_mps_msg;
//        margin_mps_msg.header.frame_id = "world";
//        margin_mps_msg.header.seq = seq;
//        margin_mps_msg.header.stamp.fromSec(timestamp);

//        for(auto& it : margin_mps) {
//            geometry_msgs::Point32 p;
//            p.x = it(0);
//            p.y = it(1);
//            p.z = it(2);
//            margin_mps_msg.points.push_back(p);
//        }
//        pub_margin_mps.publish(margin_mps_msg);
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
