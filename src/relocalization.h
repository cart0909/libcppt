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
        int64_t loop_index = -1;
        Sophus::SO3d pnp_q_old_cur;
        Eigen::Vector3d pnp_p_old_cur;
        double pnp_yaw_old_cur;
        cv::Mat matched_img;
    };
    SMART_PTR(Frame)

    Relocalization(const std::string& voc_filename, const std::string& brief_pattern_file,
                   CameraPtr camera_, const Sophus::SO3d& q_bc_, const Eigen::Vector3d& p_bc_);

    inline void SubRelocTwc(std::function<void(double, const Sophus::SE3d&)> callback) {
        pub_reloc_Twc.emplace_back(callback);
    }

    inline void SubAddRelocPath(std::function<void(const Sophus::SE3d&)> callback) {
        pub_add_reloc_path.emplace_back(callback);
    }

    inline void SubUpdateRelocPath(std::function<void(const std::vector<Sophus::SE3d>&)> callback) {
        pub_update_reloc_path.emplace_back(callback);
    }

    inline void SubRelocImg(std::function<void(const cv::Mat&)> callback) {
        pub_reloc_img.emplace_back(callback);
    }

    inline void SubLoopEdge(std::function<void(const std::pair<uint64_t, uint64_t>&)> callback) {
        pub_loop_edge.emplace_back(callback);
    }

    void ProcessFrame(FramePtr frame);
    void UpdateVIOPose(double timestamp, const Sophus::SE3d& T_viow_c);
    Sophus::SE3d ShiftPoseWorld(const Sophus::SE3d& Tviow);
    Eigen::Vector3d ShiftVectorWorld(const Eigen::Vector3d& Vviow);
private:
    int64_t DetectLoop(FramePtr frame);
    bool FindMatchesAndSolvePnP(FramePtr old_frame, FramePtr frame);
    void Optimize4DoF();

    uint64_t next_frame_id = 0;

    std::thread pose_graph_thread;
    BriefDatabase db;
    std::mutex mtx_frame_database;
    std::vector<FramePtr> v_frame_database;
    std::shared_ptr<BriefVocabulary> voc;

    DVision::BRIEF brief_extractor[2];
    CameraPtr camera;

    Sophus::SO3d q_bc;
    Eigen::Vector3d p_bc;

    std::mutex mtx_optimize_buffer;
    std::condition_variable cv_optimize_buffer;
    std::vector<int64_t> v_optimize_buffer;

    std::mutex mtx_w_viow;
    Eigen::Vector3d p_w_viow;
    Sophus::SO3d q_w_viow;

    std::vector<std::function<void(double, const Sophus::SE3d&)>> pub_reloc_Twc;
    std::vector<std::function<void(const cv::Mat&)>> pub_reloc_img;

    std::mutex mtx_reloc_path;
    std::vector<std::function<void(const Sophus::SE3d&)>> pub_add_reloc_path;
    std::vector<std::function<void(const std::vector<Sophus::SE3d>&)>> pub_update_reloc_path;
    std::vector<std::function<void(const std::pair<uint64_t, uint64_t>&)>> pub_loop_edge;
};
SMART_PTR(Relocalization)
