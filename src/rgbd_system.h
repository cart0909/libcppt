#pragma once
#include "system.h"

class RGBDSystem : public System {
public:
    RGBDSystem(const std::string& config_file);
    ~RGBDSystem();
private:
    void InitCameraParameters() override;
    void BackEndProcess() override;
};
SMART_PTR(RGBDSystem)
