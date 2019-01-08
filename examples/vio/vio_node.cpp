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
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
// catch ctrl+c signal
#include <signal.h>

#include "ros_utility.h"
#include "src/CameraPoseVisualization.h"
#include "feature_tracker.h"
#include "stereo_matcher.h"
#include "config_loader.h"
using namespace std;
using namespace message_filters;
using namespace sensor_msgs;

class Node {
public:
    using Measurements = vector<pair<pair<ImageConstPtr, ImageConstPtr>, vector<ImuConstPtr>>>;
    Node() : camera_pose_visual(1, 0, 0, 1)
    {
        camera_pose_visual.setScale(0.3);
        t_system = std::thread(&Node::SystemThread, this);
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

        ConfigLoader::Param param = ConfigLoader::Load(config_file);

        CameraPtr camera_master, camera_slave;

        camera_master = CameraPtr(new Pinhole(param.width[0], param.height[0], param.intrinsic_master[0][0],
                param.intrinsic_master[0][1], param.intrinsic_master[0][2], param.intrinsic_master[0][3],
                param.distortion_master[0][0], param.distortion_master[0][1], param.distortion_master[0][2],
                param.distortion_master[0][3]));

        camera_slave = CameraPtr(new Pinhole(param.width[0], param.height[0], param.intrinsic_slave[0][0],
                param.intrinsic_slave[0][1], param.intrinsic_slave[0][2], param.intrinsic_slave[0][3],
                param.distortion_slave[0][0], param.distortion_slave[0][1], param.distortion_slave[0][2],
                param.distortion_slave[0][3]));


        feature_tracker = std::make_shared<FeatureTracker>(camera_master);
        stereo_matcher = std::make_shared<StereoMatcher>(camera_master, camera_slave, param.p_rl[0], param.q_rl[0]);
        fs.release();
    }

    void ImageCallback(const ImageConstPtr& img_msg, const ImageConstPtr& img_r_msg) {
        unique_lock<mutex> lock(m_buf);
        img_buf.emplace(img_msg, img_r_msg);
        cv_system.notify_one();
    }

    void ImuCallback(const ImuConstPtr& imu_msg) {
        unique_lock<mutex> lock(m_buf);
        imu_buf.emplace(imu_msg);
        cv_system.notify_one();
    }

    Measurements GetMeasurements() {
        // The buffer mutex is locked before this function be called.
        Measurements measurements;

        while (1) {
            if (imu_buf.empty() || img_buf.empty())
                return measurements;

            double img_ts = img_buf.front().first->header.stamp.toSec();
            // catch the imu data before image_timestamp
            // ---------------^-----------^ image
            //                f           f+1
            // --x--x--x--x--x--x--x--x--x- imu
            //   f                       b
            // --o--o--o--o--o^-?---------- collect data in frame f

            // if ts(imu(b)) < ts(img(f)), wait imu data
            if (imu_buf.back()->header.stamp.toSec() < img_ts) {
                return measurements;
            }
            // if ts(imu(f)) > ts(img(f)), img data faster than imu data, drop the img(f)
            if (imu_buf.front()->header.stamp.toSec() > img_ts) {
                img_buf.pop();
                continue;
            }

            pair<ImageConstPtr, ImageConstPtr> img_msg = img_buf.front();
            img_buf.pop();

            vector<ImuConstPtr> IMUs;
            while (imu_buf.front()->header.stamp.toSec() < img_ts) {
                IMUs.emplace_back(imu_buf.front());
                imu_buf.pop();
            }
            // IMUs.emplace_back(imu_buf.front()); // ??
            measurements.emplace_back(img_msg, IMUs);
        }
    }

    void SystemThread() {
        while(1) {
            Measurements measurements;
            std::unique_lock<std::mutex> lock(m_buf);
            cv_system.wait(lock, [&] {
                return (measurements = GetMeasurements()).size() != 0;
            });
            lock.unlock();

            // TODO
            for(auto& meas : measurements) {
                ImageConstPtr img_msg = meas.first.first;
                ImageConstPtr img_msg_right = meas.first.second;
                vector<ImuConstPtr>& v_imu_msg = meas.second;
                double timestamp = img_msg->header.stamp.toSec();
                cv::Mat img_left, img_right;
                img_left = cv_bridge::toCvCopy(img_msg, "mono8")->image;
                img_right = cv_bridge::toCvCopy(img_msg_right, "mono8")->image;

                static bool first_frame = true;
                FeatureTracker::FramePtr feat_frame;
                StereoMatcher::FramePtr stereo_frame;
                if(first_frame) {
                    feat_frame = feature_tracker->InitFirstFrame(img_left, timestamp);
                    first_frame = false;
                }
                else {
                    feat_frame = feature_tracker->Process(img_left, timestamp);
                }

                stereo_frame = stereo_matcher->Process(feat_frame, img_right);

                cv::Mat result, result_r;
                cv::cvtColor(feat_frame->img, result, CV_GRAY2BGR);
                cv::cvtColor(stereo_frame->img_r, result_r, CV_GRAY2BGR);

                for(auto& pt : feat_frame->pt) {
                    cv::circle(result, pt, 4, cv::Scalar(0, 255, 0), -1);
                }

                for(auto& pt : stereo_frame->pt_r) {
                    if(pt.x == -1)
                        continue;
                    cv::circle(result_r, pt, 4, cv::Scalar(0, 255, 0), -1);
                }

                cv::hconcat(result, result_r, result);

                PubTrackImg(result, feat_frame->id, feat_frame->timestamp);
//                vector<ImuData> v_imu_data;

//                for(auto& it : v_imu_msg)
//                    v_imu_data.emplace_back(it->linear_acceleration.x, it->linear_acceleration.y, it->linear_acceleration.z,
//                                            it->angular_velocity.x,it->angular_velocity.y,it->angular_velocity.z,
//                                            it->header.stamp.toSec());

//                mpSystem->Process(img_left, img_right, timestamp, v_imu_data);

//                auto& frame = mpSystem->mpLastFrame;
//                PubFeatureImg(frame);
            }
        }
    }


    void PubTrackImg(const cv::Mat& track_img, uint64_t seq, double timestamp) {
        cv_bridge::CvImage track_cvimage;
        track_cvimage.header.seq = seq;
        track_cvimage.header.frame_id = "world";
        track_cvimage.header.stamp.fromSec(timestamp);
        track_cvimage.image = track_img;
        track_cvimage.encoding = sensor_msgs::image_encodings::BGR8;
        pub_track_img.publish(track_cvimage.toImageMsg());
    }

    void PubSlidingWindow(const std::vector<Sophus::SE3d>& v_Twc,
                          const Eigen::VecVector3d& v_x3Dw) {
        if(v_Twc.empty())
            return;

        static Sophus::SE3d Tglw = Sophus::SE3d::rotX(-M_PI/2);

        { // print keyframe point
            visualization_msgs::Marker key_poses;
            key_poses.header.frame_id = "world";
            key_poses.ns = "keyframes";
            key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
            key_poses.action = visualization_msgs::Marker::ADD;
            key_poses.pose.orientation.w = 1.0;
            key_poses.lifetime = ros::Duration();

            key_poses.id = 0;
            key_poses.scale.x = 0.05;
            key_poses.scale.y = 0.05;
            key_poses.scale.z = 0.05;
            key_poses.color.r = 1.0;
            key_poses.color.a = 1.0;

            for(auto& Twc : v_Twc) {
                Eigen::Vector3d twc = (Tglw*Twc).translation();
                geometry_msgs::Point pose_marker;
                pose_marker.x = twc(0);
                pose_marker.y = twc(1);
                pose_marker.z = twc(2);
                key_poses.points.emplace_back(pose_marker);
            }
            pub_keyframes.publish(key_poses);
        }

        { // print map point
            visualization_msgs::Marker msgs_points;
            msgs_points.header.frame_id = "world";
            msgs_points.ns = "mappoint";
            msgs_points.type = visualization_msgs::Marker::SPHERE_LIST;
            msgs_points.action = visualization_msgs::Marker::ADD;
            msgs_points.pose.orientation.w = 1.0;
            msgs_points.lifetime = ros::Duration();

            msgs_points.id = 0;
            msgs_points.scale.x = 0.01;
            msgs_points.scale.y = 0.01;
            msgs_points.scale.z = 0.01;
            msgs_points.color.g = 1.0;
            msgs_points.color.a = 1.0;

            for(auto& x3Dw : v_x3Dw) {
                geometry_msgs::Point point_marker;
                Eigen::Vector3d X = Tglw * x3Dw;
                point_marker.x = X(0);
                point_marker.y = X(1);
                point_marker.z = X(2);
                msgs_points.points.emplace_back(point_marker);
            }
            pub_mappoints.publish(msgs_points);
        }
    }

    void PubCurPose(const Sophus::SE3d& Twc, double timestamp) {
        static Sophus::SE3d Tglw = Sophus::SE3d::rotX(-M_PI/2);
        // public latest frame
        // path
        Sophus::SE3d Tglc = Tglw * Twc;
        Eigen::Vector3d twc = Tglc.translation();
        Eigen::Quaterniond qwc = Tglc.so3().unit_quaternion();
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
        path.poses.emplace_back(pose_stamped);
        pub_path.publish(path);

        // camera pose
        camera_pose_visual.reset();
        camera_pose_visual.add_pose(twc, qwc);
        camera_pose_visual.publish_by(pub_camera_pose, path.header);
    }

    string imu_topic;
    string img_topic[2];
    string log_filename;

    mutex m_buf;
    queue<ImuConstPtr> imu_buf;
    queue<pair<ImageConstPtr, ImageConstPtr>> img_buf;

    condition_variable cv_system;
    thread t_system;

    FeatureTrackerPtr feature_tracker;
    StereoMatcherPtr stereo_matcher;

    ros::Publisher pub_track_img;
    ros::Publisher pub_track_img_r;

    ros::Publisher pub_path;
    nav_msgs::Path path;
    ros::Publisher pub_camera_pose;
    ros::Publisher pub_keyframes;
    ros::Publisher pub_mappoints;

    CameraPoseVisualization camera_pose_visual;
};

// global variable
Node node;
void sigint_handler(int s) {
    ROS_INFO_STREAM("logging trajectory to the file");
    std::ofstream fout(node.log_filename);
    if(!fout.is_open())
        exit(1);

    for(auto& pose : node.path.poses) {
        fout << pose.header.stamp << " ";
        fout << pose.pose.position.x << " " << pose.pose.position.y << " " <<
                pose.pose.position.z << " ";
        fout << pose.pose.orientation.x << " " << pose.pose.orientation.y << " "
             << pose.pose.orientation.z << " " << pose.pose.orientation.w << std::endl;
    }

    exit(1);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "cppt_vio", ros::init_options::NoSigintHandler);
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    signal(SIGINT, sigint_handler);

    node.ReadFromNodeHandle(nh);

    message_filters::Subscriber<Image> sub_img[2] {{nh, node.img_topic[0], 100},
                                                   {nh, node.img_topic[1], 100}};
    TimeSynchronizer<Image, Image> sync(sub_img[0], sub_img[1], 100);
    sync.registerCallback(boost::bind(&Node::ImageCallback, &node, _1, _2));

    ros::Subscriber sub_imu = nh.subscribe(node.imu_topic, 2000, &Node::ImuCallback, &node,
                                           ros::TransportHints().tcpNoDelay());

    node.pub_track_img = nh.advertise<sensor_msgs::Image>("feature_img", 1000);
    node.pub_track_img_r = nh.advertise<sensor_msgs::Image>("feature_img_r", 1000);
    node.pub_path = nh.advertise<nav_msgs::Path>("path", 1000);
    node.path.header.frame_id = "world";
    node.pub_camera_pose = nh.advertise<visualization_msgs::MarkerArray>("camera_pose", 1000);
    node.pub_keyframes = nh.advertise<visualization_msgs::Marker>("keyframes", 1000);
    node.pub_mappoints = nh.advertise<visualization_msgs::Marker>("mappoints", 1000);
    ROS_INFO_STREAM("Player is ready.");

    ros::spin();
    return 0;
}
