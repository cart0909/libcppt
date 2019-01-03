#include "backend.h"

BackEnd::BackEnd(SimpleStereoCamPtr camera_, SlidingWindowPtr sliding_window_)
    : camera(camera_), sliding_window(sliding_window_) {}

void BackEnd::Process() {
    while(1) {
        std::this_thread::yield();
    }
}
