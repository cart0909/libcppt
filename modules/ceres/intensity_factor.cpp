#include "intensity_factor.h"

const int IntensityFactor::patch_halfsize = 2;
const int IntensityFactor::patch_size = 2 * IntensityFactor::patch_halfsize;
const int IntensityFactor::patch_area = IntensityFactor::patch_area * IntensityFactor::patch_area;

IntensityFactor::IntensityFactor(PinholeCameraConstPtr camera, const cv::Mat& img_ref,
                const cv::Mat& img_cur, const Eigen::Vector3d& _x3Dr,
                const Eigen::Vector2d& uv_r,
                int level)
: mLevel(level), x3Dr(_x3Dr), mImgRef(img_ref), mImgCur(img_cur), mpCamera(camera)
{
    const int border = patch_halfsize + 1;
    if(uv_r(0) - border < 0 || uv_r(1) - border < 0 || uv_r(0) + border >= img_ref.cols ||
            uv_r(1) + border >= img_ref.rows)
        assert(0);

    const int stride = mImgRef.cols; // mImgRef.step / 1
    const double scale = 1.0f / (1 << mLevel);
    const double u_ref = uv_r(0) * scale;
    const double v_ref = uv_r(1) * scale;
    const int u_ref_i = std::floor(u_ref);
    const int v_ref_i = std::floor(v_ref);

    Eigen::Matrix<double, 2, 3> proj_jac;
    Eigen::Vector2d uv_proj;
    camera->Project(x3Dr, uv_proj, proj_jac);
    Eigen::Matrix<double, 3, 6> frame_jac;
    frame_jac << Eigen::Matrix3d::Identity(), -Sophus::SO3d::hat(x3Dr);
    Eigen::Matrix<double, 2, 6> proj_frame_jac = proj_jac * frame_jac;

    const double subpix_u_ref = u_ref - u_ref_i;
    const double subpix_v_ref = v_ref - v_ref_i;
    const double w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
    const double w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
    const double w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
    const double w_ref_br = subpix_u_ref * subpix_v_ref;

    double* ref_patch_ptr = mRefPatch.data();

    for(int y = 0; y < patch_size; ++y) {
        int vbegin = v_ref_i - patch_size;
        int ubegin = u_ref_i - patch_size;
        uchar* ref_img_ptr = mImgRef.data + (vbegin + y) * stride + ubegin;
        for(int x = 0; x < patch_size; ++x, ++ref_img_ptr, ++ref_patch_ptr) {
            // bilinear intensity
            *ref_patch_ptr = w_ref_tl * ref_img_ptr[0] + w_ref_tr * ref_img_ptr[1] +
                    w_ref_bl * ref_img_ptr[stride] + w_ref_br * ref_img_ptr[stride + 1];

            // pixel graident
            double dx = 0.5 *
                    ((w_ref_tl * ref_img_ptr[1] + w_ref_tr * ref_img_ptr[2] +
                      w_ref_bl * ref_img_ptr[stride+1] + w_ref_br * ref_img_ptr[stride+2]) -
                     (w_ref_tl * ref_img_ptr[-1] + w_ref_tr * ref_img_ptr[0] +
                      w_ref_bl * ref_img_ptr[stride-1] + w_ref_br * ref_img_ptr[stride]));
            double dy = 0.5 *
                    ((w_ref_tl * ref_img_ptr[stride] + w_ref_tr * ref_img_ptr[stride+1] +
                      w_ref_bl * ref_img_ptr[stride*2] + w_ref_br * ref_img_ptr[stride*2+1]) -
                     (w_ref_tl * ref_img_ptr[-stride] + w_ref_tr * ref_img_ptr[-stride+1] +
                      w_ref_bl * ref_img_ptr[0] + w_ref_br * ref_img_ptr[1]));
            mJacobian.row(x + y * patch_size) =
                    (dx * proj_frame_jac.row(0) + dy * proj_frame_jac.row(1)) * scale;
        }
    }
}

bool IntensityFactor::Evaluate(double const* const* parameters_raw,
                      double* residuals_raw, double** jacobian_raw) const {
    const int stride = mImgCur.cols;
    const int border = patch_halfsize + 1;
    const double scale = 1.0 / (1 << mLevel);
    Eigen::Map<const Sophus::SE3d> Tcr(parameters_raw[0]);
    Eigen::Vector3d x3Dc = Tcr * x3Dr;
    Eigen::Vector2d uv_c;
    mpCamera->Project(x3Dc, uv_c);
    uv_c *= scale;

    const double u_cur = uv_c(0);
    const double v_cur = uv_c(1);
    const int u_cur_i = std::floor(u_cur);
    const int v_cur_i = std::floor(v_cur);

    if(u_cur_i - border < 0 || v_cur_i - border < 0 || u_cur_i + border >= mImgCur.cols ||
            v_cur_i + border >= mImgCur.rows)
        return false;

    const double subpix_u_cur = u_cur - u_cur_i;
    const double subpix_v_cur = v_cur - v_cur_i;
    const double w_cur_tl = (1.0f - subpix_u_cur) * (1.0f - subpix_v_cur);
    const double w_cur_tr = subpix_u_cur * (1.0f - subpix_v_cur);
    const double w_cur_bl = (1.0f - subpix_u_cur) * subpix_v_cur;
    const double w_cur_br = subpix_u_cur * subpix_v_cur;

    for(int y = 0; y < patch_size; ++y) {
        int vbegin = v_cur_i - patch_size;
        int ubegin = u_cur_i - patch_size;
        uchar* cur_img_ptr = mImgCur.data + (vbegin + y) * stride + ubegin;
        for(int x = 0; x < patch_size; ++x, ++cur_img_ptr) {
            double intensity_cur = w_cur_tl * cur_img_ptr[0] + w_cur_tr * cur_img_ptr[1] +
                    w_cur_bl * cur_img_ptr[stride] + w_cur_br * cur_img_ptr[stride + 1];
            int idx = x + y * patch_size;
            residuals_raw[idx] = mRefPatch(idx) - intensity_cur;
        }
    }

    if(jacobian_raw && jacobian_raw[0]) {
        Eigen::Map<Eigen::Matrix<double, 16, 7, Eigen::RowMajor>> jacobian(jacobian_raw[0]);
        jacobian.block<16, 1>(15, 0).setZero();
        jacobian.block<16, 6>(0, 0) = mJacobian;
    }

    return true;
}
