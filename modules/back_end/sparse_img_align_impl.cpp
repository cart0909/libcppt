#include "sparse_img_align_impl.h"

const int SparseImgAlignImpl::patch_halfsize = 2;
const int SparseImgAlignImpl::patch_size = 2 * SparseImgAlignImpl::patch_halfsize;
const int SparseImgAlignImpl::patch_area = SparseImgAlignImpl::patch_size * SparseImgAlignImpl::patch_size ;
