#pragma once
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "camera_model/pinhole_camera.h"
#include "front_end/utility.h"

class SparseImgAlignImpl {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Matrix16_6f = Eigen::Matrix<float, 16, 6>;
    using Vector16f = Eigen::Matrix<float, 16, 1>;
    using VecVector16f = std::vector<Vector16f, Eigen::aligned_allocator<Vector16f>>;
    using VecJacobianf = std::vector<Matrix16_6f, Eigen::aligned_allocator<Matrix16_6f>>;

    SparseImgAlignImpl(const PinholeCameraPtr& camera, int max_iter)
        : mMaxIter(max_iter), mpCamera(camera) {}

    void Run(int level, const cv::Mat& img_ref, const cv::Mat& img_cur,
             const VecVector2d& ref_pts, const VecVector3d& v_x3Dr,
             Sophus::SE3f& Tcr)
    {
        PreComputeJacobianAndRefPatch(level, img_ref, ref_pts, v_x3Dr);

        for(int iter = 0; iter < mMaxIter; ++iter) {

        }
    }

    void PreComputeJacobianAndRefPatch(int level, const cv::Mat& img_ref,
                                       const VecVector2d& ref_pts, const VecVector3d& v_x3Dr) {
        int num_pts = ref_pts.size();
        mvJacobian.resize(num_pts);
        mvRefPatch.resize(num_pts);
        mbAvailable.resize(num_pts, false);
        const int border = patch_halfsize + 1;
        const int stride = img_ref.cols;
        const float scale = 1.0f / (1 << level);

        for(int i = 0; i < num_pts; ++i) {
            auto& uv_r = ref_pts[i];
            auto& x3Dr = v_x3Dr[i];
            const float u_ref = uv_r(0) * scale;
            const float v_ref = uv_r(1) * scale;
            const int u_ref_i = std::floor(u_ref);
            const int v_ref_i = std::floor(v_ref);

            if(u_ref_i - border < 0 || v_ref_i - border < 0
               || u_ref_i + border >= img_ref.cols || v_ref_i + border >= img_ref.rows)
                continue;
            mbAvailable[i] = true;

            Eigen::Matrix<double, 2, 3> proj_jac;
            Eigen::Vector2d uv_proj;
            mpCamera->Project(x3Dr, uv_proj, proj_jac);
            Eigen::Matrix<double, 3, 6> frame_jac;
            frame_jac << Eigen::Matrix3d::Identity(), -Sophus::SO3d::hat(x3Dr);
            Eigen::Matrix<float, 2, 6> proj_frame_jac = proj_jac.cast<float>() * frame_jac.cast<float>();

            const float subpix_u_ref = u_ref - u_ref_i;
            const float subpix_v_ref = v_ref - v_ref_i;
            const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
            const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
            const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;

            float* ref_patch_ptr = mvRefPatch[i].data();

            for(int y = 0; y < patch_size; ++y) {
                int vbegin = v_ref_i - patch_size;
                int ubegin = u_ref_i - patch_size;
                uchar* ref_img_ptr = img_ref.data + (vbegin + y) * stride + ubegin;
                for(int x = 0; x < patch_size; ++x, ++ref_img_ptr, ++ref_patch_ptr) {
                    // bilinear intensity
                    *ref_patch_ptr = w_ref_tl * ref_img_ptr[0] + w_ref_tr * ref_img_ptr[1] +
                            w_ref_bl * ref_img_ptr[stride] + w_ref_br * ref_img_ptr[stride + 1];

                    // pixel graident
                    float dx = 0.5 *
                            ((w_ref_tl * ref_img_ptr[1] + w_ref_tr * ref_img_ptr[2] +
                              w_ref_bl * ref_img_ptr[stride+1] + w_ref_br * ref_img_ptr[stride+2]) -
                             (w_ref_tl * ref_img_ptr[-1] + w_ref_tr * ref_img_ptr[0] +
                              w_ref_bl * ref_img_ptr[stride-1] + w_ref_br * ref_img_ptr[stride]));
                    float dy = 0.5 *
                            ((w_ref_tl * ref_img_ptr[stride] + w_ref_tr * ref_img_ptr[stride+1] +
                              w_ref_bl * ref_img_ptr[stride*2] + w_ref_br * ref_img_ptr[stride*2+1]) -
                             (w_ref_tl * ref_img_ptr[-stride] + w_ref_tr * ref_img_ptr[-stride+1] +
                              w_ref_bl * ref_img_ptr[0] + w_ref_br * ref_img_ptr[1]));
                    mvJacobian[i].row(x + y * patch_size) =
                            (dx * proj_frame_jac.row(0) + dy * proj_frame_jac.row(1)) * scale;
                }
            }
        }
    }

    void ComputeResidual();

private:
    static const int patch_halfsize;
    static const int patch_size;
    static const int patch_area;
    int mMaxIter; // 10?

    PinholeCameraPtr mpCamera;
    VecVector16f mvRefPatch;
    VecJacobianf mvJacobian;
    std::vector<bool> mbAvailable;
};

using SparseImgAlignImplPtr = std::shared_ptr<SparseImgAlignImpl>;
using SparseImgAlignImplConstPtr = std::shared_ptr<const SparseImgAlignImpl>;
