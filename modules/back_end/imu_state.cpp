#include "imu_state.h"

uint64_t ImuState::g_next_id = 0;
Eigen::Vector3d ImuState::g_gravity;
