#include "util_datatype.h"

bool InBorder(const cv::Point2f& pt, int width, int height) {
    const int border_size = 1;
    int img_x = std::round(pt.x);
    int img_y = std::round(pt.y);
    if(img_x - border_size < 0 || img_y - border_size <= 0 || img_x + border_size >= width ||
            img_y + border_size >= height)
        return false;
    return true;
}
