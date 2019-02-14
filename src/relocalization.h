#pragma once
#include <mutex>
#include <thread>
#include <memory>
#include <condition_variable>
#include "DBoW2/DBoW2/DBoW2.h"
#include "DBoW2/DVision/DVision.h"
#include "DBoW2/DBoW2/TemplatedDatabase.h"
#include "DBoW2/DBoW2/TemplatedVocabulary.h"
#include "util.h"

class Relocalization {
public:
    struct Frame {
        uint64_t frame_id;
        double timestamp;
        cv::Mat img;
        std::vector<uint64_t> v_pt_id;
        std::vector<cv::Point2f> v_pt_2d;
        std::vector<cv::Point3f> v_pt_3d;
        std::vector<DVision::BRIEF::bitset> v_descriptor;

        std::vector<cv::KeyPoint> v_extra_keypoint;
        std::vector<DVision::BRIEF::bitset> v_extra_descriptor;
        Sophus::SE3d q_wc;
        Eigen::Vector3d p_wc;
    };
    SMART_PTR(Frame)

    Relocalization(const std::string& voc_filename) {
        voc = std::make_shared<BriefVocabulary>(voc_filename);
        db.setVocabulary(*voc, false, 0);
        detect_loop_thread = std::thread(&Relocalization::DetectLoop, this);
    }

    void PushFrame(FramePtr frame) {
        m_frame_buffer.lock();
        v_frame_buffer.emplace_back(frame);
        m_frame_buffer.unlock();
        cv_frame_buffer.notify_one();
    }

    void DetectLoop() {
        while(1) {
            std::vector<FramePtr> v_frames;
            std::unique_lock<std::mutex> lock(m_frame_buffer);
            cv_frame_buffer.wait(lock, [&] {
               v_frames = v_frame_buffer;
               v_frame_buffer.clear();
               return !v_frames.empty();
            });
        }
    }

    void Optimize4DoF() {
        while(1) {

        }
    }

private:
    std::mutex m_frame_buffer;
    std::condition_variable cv_frame_buffer;
    std::vector<FramePtr> v_frame_buffer;

    std::thread detect_loop_thread, pose_graph_thread;
    BriefDatabase db;
    std::shared_ptr<BriefVocabulary> voc;
};
