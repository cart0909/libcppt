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
#include "ros_utility.h"
#include "system/vo_system.h"
using namespace std;
using namespace message_filters;
using namespace sensor_msgs;

class Node {
public:
    using Measurements = vector<pair<pair<ImageConstPtr, ImageConstPtr>, vector<ImuConstPtr>>>;
    Node() {
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

        mpSystem = std::make_shared<VOSystem>(config_file);
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
                auto& img_msg = meas.first.first;
                auto& img_msg_right = meas.first.second;
                double timestamp = img_msg->header.stamp.toSec();
                cv::Mat img_left, img_right;
                img_left = cv_bridge::toCvCopy(img_msg, "mono8")->image;
                img_right = cv_bridge::toCvCopy(img_msg_right, "mono8")->image;

                mpSystem->Process(img_left, img_right, timestamp);

                auto& frame = mpSystem->mpLastFrame;
                PubFeatureImg(frame);
            }
        }
    }

    void PubFeatureImg(const FramePtr& frame) {
        cv::Mat feature_img, feature_img_r;
        cv::cvtColor(frame->mImgL, feature_img, CV_GRAY2BGR);
        cv::cvtColor(frame->mImgR, feature_img_r, CV_GRAY2BGR);
        for(int i = 0, n = frame->mv_uv.size(); i < n; ++i) {
            auto& pt = frame->mv_uv[i];
            auto& ur = frame->mv_ur[i];
            cv::circle(feature_img, pt, 2, cv::Scalar(0, 255, 0), -1);
            if(ur != -1)
                cv::circle(feature_img_r, cv::Point(ur, pt.y), 2, cv::Scalar(0, 255, 0), -1);
        }

        cv_bridge::CvImage feature_img_msg_l, feature_img_msg_r;
        feature_img_msg_l.header.seq = frame->mFrameID;
        feature_img_msg_l.header.frame_id = "world";
        feature_img_msg_l.header.stamp.fromSec(frame->mTimeStamp);
        feature_img_msg_l.image = feature_img;
        feature_img_msg_l.encoding = sensor_msgs::image_encodings::RGB8;
        pub_track_img.publish(feature_img_msg_l.toImageMsg());

        feature_img_msg_r.header.seq = frame->mFrameID;
        feature_img_msg_r.header.frame_id = "world";
        feature_img_msg_r.header.stamp.fromSec(frame->mTimeStamp);
        feature_img_msg_r.image = feature_img_r;
        feature_img_msg_r.encoding = sensor_msgs::image_encodings::RGB8;
        pub_track_img_r.publish(feature_img_msg_r.toImageMsg());
    }

    string imu_topic;
    string img_topic[2];

    mutex m_buf;
    queue<ImuConstPtr> imu_buf;
    queue<pair<ImageConstPtr, ImageConstPtr>> img_buf;

    condition_variable cv_system;
    thread t_system;

    VOSystemPtr mpSystem;

    ros::Publisher pub_track_img;
    ros::Publisher pub_track_img_r;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "cppt_player");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    Node node;
    node.ReadFromNodeHandle(nh);

    message_filters::Subscriber<Image> sub_img[2] {{nh, node.img_topic[0], 100},
                                                   {nh, node.img_topic[1], 100}};
    TimeSynchronizer<Image, Image> sync(sub_img[0], sub_img[1], 100);
    sync.registerCallback(boost::bind(&Node::ImageCallback, &node, _1, _2));

    ros::Subscriber sub_imu = nh.subscribe(node.imu_topic, 2000, &Node::ImuCallback, &node,
                                           ros::TransportHints().tcpNoDelay());

    node.pub_track_img = nh.advertise<sensor_msgs::Image>("feature_img", 1000);
    node.pub_track_img_r = nh.advertise<sensor_msgs::Image>("feature_img_r", 1000);

    ROS_INFO_STREAM("Player is ready.");

    ros::spin();
    return 0;
}
