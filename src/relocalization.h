#pragma once
#include <mutex>
#include <thread>
#include <memory>
#include <condition_variable>
#include "3rdParty/DBoW2/DBoW2/DBoW2.h"
#include "3rdParty/DBoW2/DVision/DVision.h"
#include "3rdParty/DBoW2/DBoW2/TemplatedDatabase.h"
#include "3rdParty/DBoW2/DBoW2/TemplatedVocabulary.h"
#include "util.h"
#include "camera.h"

class Relocalization {
public:
    struct Frame {
        uint64_t frame_id;
        double timestamp;
        cv::Mat img;
        cv::Mat compressed_img;
        std::vector<uint64_t> v_pt_id;
        std::vector<cv::Point2f> v_pt_2d_uv;
        std::vector<cv::Point2f> v_pt_2d_normal; // un vn
        std::vector<cv::Point3f> v_pt_3d; // x3Dc
        std::vector<DVision::BRIEF::bitset> v_descriptor;

        std::vector<cv::Point2f> v_extra_pt_2d_uv;
        std::vector<cv::Point2f> v_extra_pt_2d_normal;
        std::vector<DVision::BRIEF::bitset> v_extra_descriptor;
        Sophus::SO3d vio_q_wb, q_wb;
        Eigen::Vector3d vio_p_wb, p_wb;

        bool has_loop = false;
        int loop_index = -1;
        Sophus::SO3d pnp_q_old_cur;
        Eigen::Vector3d pnp_p_old_cur;
        double pnp_yaw_old_cur;
        cv::Mat matched_img;
    };
    SMART_PTR(Frame)

    Relocalization(const std::string& voc_filename, const std::string& brief_pattern_file,
                   CameraPtr camera_, const Sophus::SO3d& q_bc_, const Eigen::Vector3d& p_bc_);

    void PushFrame(FramePtr frame);

private:
    void Process();
    void ProcessFrame(FramePtr frame);
    int64_t DetectLoop(FramePtr frame);
    bool FindMatchesAndSolvePnP(FramePtr old_frame, FramePtr frame);
    void Optimize4DoF();

    uint64_t next_frame_id = 0;
    std::mutex m_frame_buffer;
    std::condition_variable cv_frame_buffer;
    std::vector<FramePtr> v_frame_buffer;

    std::thread detect_loop_thread, pose_graph_thread;
    BriefDatabase db;
    std::vector<FramePtr> v_frame_database;
    std::shared_ptr<BriefVocabulary> voc;

    DVision::BRIEF brief_extractor[2];
    CameraPtr camera;

    Sophus::SO3d q_bc;
    Eigen::Vector3d p_bc;

    std::mutex mtx_optimize_buffer;
    std::condition_variable cv_optmize_buffer;
//    std::vector<uint64_t>
};
SMART_PTR(Relocalization)
