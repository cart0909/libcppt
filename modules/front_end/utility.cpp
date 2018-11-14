#include "utility.h"

namespace Utility {
ImagePyr Pyramid(const cv::Mat& img, int num_levels) {
    assert(num_levels > 1);
    ImagePyr image_pyr;
    image_pyr.resize(num_levels);
    image_pyr[0] = img;

    for(int i = 1; i < num_levels; ++i) {
        cv::resize(image_pyr[0], image_pyr[1], image_pyr[0].size()/2);
    }
    return image_pyr;
}
}
