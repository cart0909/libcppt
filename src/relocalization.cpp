#include "relocalization.h"
#include <glog/logging.h>

Relocalization::Relocalization(const std::string& voc_filename, const std::string& brief_pattern_file,
                               CameraPtr camera_)
    : camera(camera_)
{
    voc = std::make_shared<BriefVocabulary>(voc_filename);
    db.setVocabulary(*voc, false, 0);
    detect_loop_thread = std::thread(&Relocalization::Process, this);

    // load brief pattern
    cv::FileStorage fs(brief_pattern_file, cv::FileStorage::READ);
    if(!fs.isOpened()) {
        throw std::runtime_error("Could not open the BRIEF pattern file: " + brief_pattern_file);
    }

    std::vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;

    brief_extractor[0].importPairs(x1, y1, x2, y2);
    brief_extractor[1].importPairs(x1, y1, x2, y2);
}

void Relocalization::PushFrame(FramePtr frame) {
    m_frame_buffer.lock();
    v_frame_buffer.emplace_back(frame);
    m_frame_buffer.unlock();
    cv_frame_buffer.notify_one();
}

void Relocalization::Process() {
    while(1) {
        std::vector<FramePtr> v_frames;
        std::unique_lock<std::mutex> lock(m_frame_buffer);
        cv_frame_buffer.wait(lock, [&] {
           v_frames = v_frame_buffer;
           v_frame_buffer.clear();
           return !v_frames.empty();
        });

        for(auto& frame : v_frames) {
            DetectLoop(frame);
        }
    }
}

void Relocalization::DetectLoop(FramePtr frame) {
    // re-id
    frame->frame_id = next_frame_id++;

    // extract extra fast corner
    std::vector<cv::KeyPoint> v_keypoint, v_extra_keypoint;
    cv::FAST(frame->img, v_extra_keypoint, 20, true);
    // extract feature descriptor
    for(int i = 0, n = frame->v_pt_2d_uv.size(); i < n; ++i) {
        cv::KeyPoint kp;
        kp.pt = frame->v_pt_2d_uv[i];
        v_keypoint.emplace_back(kp);
    }

    brief_extractor[0].compute(frame->img, v_keypoint, frame->v_descriptor);
    brief_extractor[1].compute(frame->img, v_extra_keypoint, frame->v_extra_descriptor);

    for(int i = 0, n = v_extra_keypoint.size(); i < n; ++i) {
        cv::Point2f pt = v_extra_keypoint[i].pt;
        frame->v_extra_pt_2d_uv.emplace_back(pt);
        Eigen::Vector3d pt_normal;
        camera->BackProject(Eigen::Vector2d(pt.x, pt.y), pt_normal);
        frame->v_extra_pt_2d_normal.emplace_back(pt_normal(0), pt_normal(1));
    }
    // end of extract feature

    // release img memory
    // for debug using resize img
    cv::resize(frame->img, frame->compressed_img, frame->img.size() / 2);
    cv::cvtColor(frame->compressed_img, frame->compressed_img, CV_GRAY2BGR);
    frame->img = cv::Mat();


    DBoW2::QueryResults ret;
    db.query(frame->v_extra_descriptor, ret, 4, frame->frame_id - 50);

    db.add(frame->v_extra_descriptor);
    v_frame_database.emplace_back(frame);

    if(frame->frame_id <= 50)
        return;

    // detect loop step 1, check bag of word similar
    // a good match with its neighbor
    // 0.05 is magic number
    // 0.015 is also magic number
    bool find_loop = false;
    int min_index = std::numeric_limits<int>::max();
    if(ret.size() > 1 && ret[0].Score > 0.05)
        for(int i = 1, n = ret.size(); i < n; ++i) {
            if(ret[i].Score > 0.015) {
                find_loop = true;
                if(ret[i].Id < min_index) {
                    min_index = ret[i].Id;
                }
            }
        }

    if(find_loop) {
        cv::imshow("ref", v_frame_database[min_index]->compressed_img);
        cv::imshow("cur", frame->compressed_img);

        LOG(INFO) << "ref: " << min_index << ", cur: " << frame->frame_id;

        cv::waitKey(1);
    }
}

void Optimize4DoF() {
    while(1) {

    }
}
