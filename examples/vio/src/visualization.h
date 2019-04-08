#pragma once
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include "system.h"
#define POSE_LOG 1
namespace vis {
void ReadFromNodeHandle(ros::NodeHandle& nh, SystemPtr system);
void PubTrackImg(double timestamp, const cv::Mat& track_img);
void PubVioPose(double timestamp, const Sophus::SE3d& Twc);
void PubRelocImg(const cv::Mat& reloc_img);
void PubRelocPose(double timestamp, const Sophus::SE3d& Twc);
void AddRelocPath(const Sophus::SE3d& Twc);
void UpdateRelocPath(const std::vector<Sophus::SE3d>& v_Twc);
void PushLoopEdgeIndex(const std::pair<uint64_t, uint64_t>& edge);
void PubMapPoint(const Eigen::VecVector3d& mps);
void PubLines(const Eigen::VecVector3d& v_Pw, const Eigen::VecVector3d& v_Qw);
void RecordPose();
}
