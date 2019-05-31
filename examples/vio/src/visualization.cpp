#include "visualization.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include "CameraPoseVisualization.h"
#include "pl_system.h"
#include "rgbd_system.h"

namespace vis {
static ros::Publisher pub_track_img;
static ros::Publisher pub_vio_pose;
static ros::Publisher pub_vio_path;
static nav_msgs::Path vio_path;
static CameraPoseVisualization vio_pose_visual(0, 1, 0, 1);

// show reloc
static nav_msgs::Path reloc_path;
static CameraPoseVisualization reloc_pose_visual(1, 0, 0, 1);
static CameraPoseVisualization loop_edge_visual(0, 1, 0, 1);
static ros::Publisher pub_reloc_path;
static ros::Publisher pub_reloc_pose;
static ros::Publisher pub_loop_edge;
static ros::Publisher pub_reloc_img;
static ros::Publisher pub_transform;
static ros::Publisher pub_reloc_Twb;
// show features
static ros::Publisher pub_mappoints;
static ros::Publisher pub_lines;

static std::mutex mtx_loop_edge_index;
static std::vector<std::pair<uint64_t, uint64_t>> v_loop_edge_index;

void ReadFromNodeHandle(ros::NodeHandle& nh, SystemPtr system) {
    pub_track_img = nh.advertise<sensor_msgs::Image>("track_img", 1000);
    pub_vio_pose = nh.advertise<visualization_msgs::MarkerArray>("vio_pose", 1000);
    pub_vio_path = nh.advertise<nav_msgs::Path>("vio_path", 1000);
    vio_path.header.frame_id = "world";
    pub_reloc_path = nh.advertise<nav_msgs::Path>("reloc_path", 1000);
    reloc_path.header.frame_id = "world";
    pub_reloc_pose = nh.advertise<visualization_msgs::MarkerArray>("reloc_pose", 1000);
    pub_loop_edge = nh.advertise<visualization_msgs::MarkerArray>("loop_edge", 1000);
    pub_reloc_img = nh.advertise<sensor_msgs::Image>("reloc_img", 1000);
    pub_mappoints = nh.advertise<sensor_msgs::PointCloud>("mappoints", 1000);
    pub_lines = nh.advertise<visualization_msgs::Marker>("lines", 1000);
    pub_transform = nh.advertise<geometry_msgs::TransformStamped>("transform", 1000);
    pub_reloc_Twb = nh.advertise<nav_msgs::Odometry>("/cppt_reloc_wb", 100); //Twb
    vio_pose_visual.setScale(0.3);

    system->SubTrackingImg(std::bind(&PubTrackImg, std::placeholders::_1,
                                     std::placeholders::_2));
    system->SubVIOTwc(std::bind(&PubVioPose, std::placeholders::_1, std::placeholders::_2));
    system->SubRelocTwc(std::bind(&PubRelocPose,  std::placeholders::_1,
                                  std::placeholders::_2));
    system->SubAddRelocPath(std::bind(&AddRelocPath, std::placeholders::_1));
    system->SubUpdateRelocPath(std::bind(&UpdateRelocPath, std::placeholders::_1));
    system->SubLoopEdge(std::bind(&PushLoopEdgeIndex, std::placeholders::_1));
    system->SubRelocImg(std::bind(&PubRelocImg, std::placeholders::_1));
    system->SubMapPoints(std::bind(&PubMapPoint, std::placeholders::_1));
    system->SubRelocTwb(std::bind(&PubRelocPoseW2B, std::placeholders::_1, std::placeholders::_2));
    PLSystemPtr pl_system = std::dynamic_pointer_cast<PLSystem>(system);
    if(pl_system) {
        pl_system->SubLines(std::bind(&PubLines, std::placeholders::_1, std::placeholders::_2));
    }
}

void PubTrackImg(double timestamp, const cv::Mat& track_img) {
    cv_bridge::CvImage cvimage;
    cvimage.header.frame_id = "world";
    cvimage.header.stamp.fromSec(timestamp);
    cvimage.image = track_img;
    cvimage.encoding = sensor_msgs::image_encodings::BGR8;
    pub_track_img.publish(cvimage.toImageMsg());
}

void PubVioPose(double timestamp, const Sophus::SE3d& Twc) {
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

void PubRelocImg(const cv::Mat& reloc_img) {
    cv_bridge::CvImage reloc_cvimg;
    reloc_cvimg.header.frame_id = "world";
    reloc_cvimg.image = reloc_img;
    reloc_cvimg.encoding = sensor_msgs::image_encodings::BGR8;
    pub_reloc_img.publish(reloc_cvimg.toImageMsg());
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

    geometry_msgs::TransformStamped transform_stamped;
    transform_stamped.header = header;
    transform_stamped.child_frame_id = "camera";
    transform_stamped.transform.rotation.w = qwc.w();
    transform_stamped.transform.rotation.x = qwc.x();
    transform_stamped.transform.rotation.y = qwc.y();
    transform_stamped.transform.rotation.z = qwc.z();

    transform_stamped.transform.translation.x = twc(0);
    transform_stamped.transform.translation.y = twc(1);
    transform_stamped.transform.translation.z = twc(2);
    pub_transform.publish(transform_stamped);

    // for dense mapping (skimap, openchisel)
    static tf::TransformBroadcaster tf_broadcaster;
    tf::StampedTransform tf_stamped_transform(
                tf::Transform(tf::Quaternion(qwc.x(), qwc.y(), qwc.z(), qwc.w()), tf::Vector3(twc(0), twc(1), twc(2))),
                header.stamp, "world", "camera");
    tf_broadcaster.sendTransform(tf_stamped_transform);
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

void PubMapPoint(const Eigen::VecVector3d& mps) {
    sensor_msgs::PointCloud mps_msg;
    mps_msg.header.frame_id = "world";

    for(auto& x3Dw : mps) {
        geometry_msgs::Point32 point_marker;
        point_marker.x = x3Dw(0);
        point_marker.y = x3Dw(1);
        point_marker.z = x3Dw(2);
        mps_msg.points.emplace_back(point_marker);
    }
    pub_mappoints.publish(mps_msg);
}

void PubRelocPoseW2B(double timestamp, const Sophus::SE3d& Twb) {
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp.fromSec(timestamp);
    Eigen::Vector3d twb = Twb.translation();
    Eigen::Quaterniond qwb = Twb.unit_quaternion();

    //Pub Odometry
    nav_msgs::Odometry odomCpptMapped;
    odomCpptMapped.header = header;
    odomCpptMapped.child_frame_id = "base_link";
    odomCpptMapped.pose.pose.orientation.x = qwb.x();
    odomCpptMapped.pose.pose.orientation.y = qwb.y();
    odomCpptMapped.pose.pose.orientation.z = qwb.z();
    odomCpptMapped.pose.pose.orientation.w = qwb.w();
    odomCpptMapped.pose.pose.position.x = twb.x();
    odomCpptMapped.pose.pose.position.y = twb.y();
    odomCpptMapped.pose.pose.position.z = twb.z();
    pub_reloc_Twb.publish(odomCpptMapped);

 }

void PubLines(const Eigen::VecVector3d& v_Pw, const Eigen::VecVector3d& v_Qw) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.ns = "lines";
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.01;
    marker.scale.y = 0.01;
    marker.scale.z = 0.01;
    marker.color.b = 1.0f;
    marker.color.g = 0.4f;
    marker.color.r = 1.0f;
    marker.color.a = 1.0f;

    for(int i = 0, n = v_Pw.size(); i < n; ++i) {
        geometry_msgs::Point Pw, Qw;
        Pw.x = v_Pw[i](0);
        Pw.y = v_Pw[i](1);
        Pw.z = v_Pw[i](2);
        Qw.x = v_Qw[i](0);
        Qw.y = v_Qw[i](1);
        Qw.z = v_Qw[i](2);
        marker.points.emplace_back(Pw);
        marker.points.emplace_back(Qw);
    }

    pub_lines.publish(marker);
}
}
