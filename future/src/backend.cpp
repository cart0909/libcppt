#include "backend.h"
#include <glog/logging.h>

BackEnd::BackEnd(double focal_length_,
                 double gyr_n, double acc_n,
                 double gyr_w, double acc_w,
                 const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
                 const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_,
                 uint window_size_)
    : focal_length(focal_length_), p_rl(p_rl_), p_bc(p_bc_), q_rl(q_rl_), q_bc(q_bc_), window_size(window_size_),
      frame_count(0), next_frame_id(0), state(NEED_INIT)
{
    gyr_noise_cov = gyr_n * gyr_n * Eigen::Matrix3d::Identity();
    acc_noise_cov = acc_n * acc_n * Eigen::Matrix3d::Identity();
    gyr_bias_cov = gyr_w * gyr_w * Eigen::Matrix3d::Identity();
    acc_bias_cov = acc_w * acc_w * Eigen::Matrix3d::Identity();

    v_frames.resize(window_size_ + 1); // 10 old kf + 1 new kf
    thread_ = std::thread(&BackEnd::Process, this);
}

BackEnd::~BackEnd() {}

void BackEnd::PushFrame(FramePtr frame) {
    m_buffer.lock();
    frame_buffer.emplace_back(frame);
    frame->id = next_frame_id++;
    m_buffer.unlock();
    cv_buffer.notify_one();
}

void BackEnd::Process() {
    while(1) {
        std::vector<FramePtr> measurements;
        std::unique_lock<std::mutex> lock(m_buffer);
        cv_buffer.wait(lock, [&] {
            measurements = std::vector<FramePtr>(frame_buffer.begin(), frame_buffer.end());
            frame_buffer.clear();
            return !measurements.empty();
        });

        for(auto& frame : measurements) {
            ProcessFrame(frame);
        }
    }
}

void BackEnd::ProcessFrame(FramePtr frame) {
    // test graph
    for(int i = 0; i < frame->pt_id.size(); ++i) {
        auto it = m_features.find(frame->pt_id[i]);
        Feature* feat = nullptr;
        if(it == m_features.end()) {
            auto result = m_features.emplace(std::make_pair(frame->pt_id[i], Feature(frame->pt_id[i], frame_count)));
            if(!result.second)
                throw std::runtime_error("m_features insert fail?");
            feat = &result.first->second;
        }
        else {
            feat = &it->second;
        }

        feat->pt_n_per_frame.emplace_back(frame->pt_normal_plane[i]);
        feat->pt_r_n_per_frame.emplace_back(frame->pt_r_normal_plane[i]);
        if(frame->pt_r_normal_plane[i](0) == -1)
            feat->num_meas += 1;
        else
            feat->num_meas += 2;
    }

    v_frames[frame_count] = frame;

    if(frame_count >= window_size) {
        // margin out old
        for(auto it = m_features.begin(); it != m_features.end();) {
            auto& feat = it->second;
            if(feat.sliding_window_id == 0) {
                // change parent
                if(feat.pt_r_n_per_frame[0](0) == -1)
                    feat.num_meas -= 1;
                else
                    feat.num_meas -= 2;

                feat.pt_n_per_frame.pop_front();
                feat.pt_r_n_per_frame.pop_front();

                if(feat.num_meas < 2) {
                    if(feat.pt_n_per_frame.empty() && feat.pt_r_n_per_frame.empty()) {
                        it = m_features.erase(it);
                    }
                    else {
                        throw std::runtime_error("feature num meas count wrong!?");
                    }
                }
            }
            else {
                // TODO
            }
        }

        v_frames.pop_front();
    }
    else {
        ++frame_count;
    }
}
