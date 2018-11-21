#pragma once
#include <memory>
#include <mutex>

class MapPoint {
public:

};

using MapPointPtr = std::shared_ptr<MapPoint>;
using MapPointConstPtr = std::shared_ptr<const MapPoint>;
