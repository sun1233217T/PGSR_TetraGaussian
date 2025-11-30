#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include "rasterizer_torch.h"

namespace {

// 简单占位 kernel：为每条射线写常量颜色。
__device__ inline bool intersect_aabb(
    const float3& o,
    const float3& d,
    const float3& bmin,
    const float3& bmax,
    float& t0,
    float& t1) {
    float tmin = 0.f;
    float tmax = 1e20f;
    for (int i = 0; i < 3; ++i) {
        float o_c = (&o.x)[i];
        float d_c = (&d.x)[i];
        float min_c = (&bmin.x)[i];
        float max_c = (&bmax.x)[i];
        if (fabsf(d_c) < 1e-12f) {
            if (o_c < min_c || o_c > max_c) return false;
            continue;
        }
        float inv = 1.0f / d_c;
        float tnear = (min_c - o_c) * inv;
        float tfar  = (max_c - o_c) * inv;
        if (tnear > tfar) {
            float tmp = tnear;
            tnear = tfar;
            tfar = tmp;
        }
        tmin = tnear > tmin ? tnear : tmin;
        tmax = tfar  < tmax ? tfar  : tmax;
        if (tmin > tmax) return false;
    }
    t0 = tmin;
    t1 = tmax;
    return true;
}

__global__ void rasterize_kernel(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ keys,
    const uint64_t* __restrict__ mask,
    float* __restrict__ out,
    int64_t num_rays,
    int64_t width,
    float3 origin,
    float voxel_size,
    int3 dims,
    int3 brick_size,
    int64_t coarse_res) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;

    float3 o = make_float3(rays_o[idx * 3 + 0], rays_o[idx * 3 + 1], rays_o[idx * 3 + 2]);
    float3 d = make_float3(rays_d[idx * 3 + 0], rays_d[idx * 3 + 1], rays_d[idx * 3 + 2]);

    // AABB of full grid
    float3 bmin = origin;
    float3 bmax = make_float3(origin.x + voxel_size * dims.x,
                              origin.y + voxel_size * dims.y,
                              origin.z + voxel_size * dims.z);
    float t0, t1;
    if (!intersect_aabb(o, d, bmin, bmax, t0, t1) || t1 < 0.f) {
        return;
    }
    if (t0 < 0.f) t0 = 0.f;

    // brick bounds helpers (per-axis, handle last brick width)
    auto brick_bounds = [&](int b, int brick, int dim, float org, float& start, float& end) {
        int v_start = b * brick;
        int v_end = v_start + brick;
        if (v_end > dim) v_end = dim;
        start = org + voxel_size * static_cast<float>(v_start);
        end = org + voxel_size * static_cast<float>(v_end);
    };
    auto t_to_exit = [&](float o_c, float d_c, float start, float end, int step_c) {
        if (fabsf(d_c) < 1e-12f || step_c == 0) return 1e20f;
        float boundary = step_c > 0 ? end : start;
        return (boundary - o_c) / d_c;
    };
    auto delta_t = [&](float d_c, float start, float end) {
        if (fabsf(d_c) < 1e-12f) return 1e20f;
        return (end - start) / fabsf(d_c);
    };

    // starting coarse idx: 先转体素索引再按整数砖划分，和 host 端一致
    auto voxel_idx = [&](float p_c, float org_c, int dim_c) {
        int v = static_cast<int>(floorf((p_c - org_c) / voxel_size));
        if (v < 0) v = 0;
        if (v >= dim_c) v = dim_c - 1;
        return v;
    };
    float3 p = make_float3(o.x + d.x * t0, o.y + d.y * t0, o.z + d.z * t0);
    int vx0 = voxel_idx(p.x, origin.x, dims.x);
    int vy0 = voxel_idx(p.y, origin.y, dims.y);
    int vz0 = voxel_idx(p.z, origin.z, dims.z);
    int cx = vx0 / brick_size.x;
    int cy = vy0 / brick_size.y;
    int cz = vz0 / brick_size.z;
    auto in_bounds = [&](int x, int y, int z) {
        return x >= 0 && y >= 0 && z >= 0 && x < coarse_res && y < coarse_res && z < coarse_res;
    };
    if (!in_bounds(cx, cy, cz)) return;

    int step_x = (d.x > 0) ? 1 : (d.x < 0 ? -1 : 0);
    int step_y = (d.y > 0) ? 1 : (d.y < 0 ? -1 : 0);
    int step_z = (d.z > 0) ? 1 : (d.z < 0 ? -1 : 0);

    float sx0, sx1, sy0, sy1, sz0, sz1;
    brick_bounds(cx, brick_size.x, dims.x, origin.x, sx0, sx1);
    brick_bounds(cy, brick_size.y, dims.y, origin.y, sy0, sy1);
    brick_bounds(cz, brick_size.z, dims.z, origin.z, sz0, sz1);

    float tMaxX = t_to_exit(o.x, d.x, sx0, sx1, step_x);
    float tMaxY = t_to_exit(o.y, d.y, sy0, sy1, step_y);
    float tMaxZ = t_to_exit(o.z, d.z, sz0, sz1, step_z);
    float tDeltaX = delta_t(d.x, sx0, sx1);
    float tDeltaY = delta_t(d.y, sy0, sy1);
    float tDeltaZ = delta_t(d.z, sz0, sz1);

    // 细体素级占位渲染：在当前 coarse brick 内逐个细体素做 AABB 相交并累加颜色
    const float eps = 1e-4f; // 边界容差，避免因浮点截断漏掉靠近砖界的体素
    while (in_bounds(cx, cy, cz) && t0 <= t1) {
        int64_t b = (static_cast<int64_t>(cx) * coarse_res + cy) * coarse_res + cz;
        int64_t off0 = offsets[b];
        int64_t off1 = offsets[b + 1];
        if (off1 > off0) {
            float brick_exit_t = fminf(tMaxX, fminf(tMaxY, tMaxZ)) + eps; // 当前 coarse brick 的射线退出时间

            // 先在 4x4x4 细分格上做一次 DDA，利用 mask 快速跳过无占据子块
            float search_start_t = t0;
            uint64_t brick_mask = mask[b];
            if (brick_mask != 0ULL) {
                float sub_size_x = voxel_size * (static_cast<float>(brick_size.x) / 4.f);
                float sub_size_y = voxel_size * (static_cast<float>(brick_size.y) / 4.f);
                float sub_size_z = voxel_size * (static_cast<float>(brick_size.z) / 4.f);

                // 当前点所在子块索引（0..3）
                float3 p_cur = make_float3(o.x + d.x * t0, o.y + d.y * t0, o.z + d.z * t0);
                int sub_x = static_cast<int>(floorf((p_cur.x - sx0) / sub_size_x));
                int sub_y = static_cast<int>(floorf((p_cur.y - sy0) / sub_size_y));
                int sub_z = static_cast<int>(floorf((p_cur.z - sz0) / sub_size_z));
                sub_x = max(0, min(3, sub_x));
                sub_y = max(0, min(3, sub_y));
                sub_z = max(0, min(3, sub_z));

                auto sub_bounds = [&](int s, float sub_size, float start_axis, float& a0, float& a1) {
                    a0 = start_axis + sub_size * static_cast<float>(s);
                    a1 = (s == 3) ? start_axis + sub_size * 4.f : a0 + sub_size;
                };
                float sx_sub0, sx_sub1, sy_sub0, sy_sub1, sz_sub0, sz_sub1;
                sub_bounds(sub_x, sub_size_x, sx0, sx_sub0, sx_sub1);
                sub_bounds(sub_y, sub_size_y, sy0, sy_sub0, sy_sub1);
                sub_bounds(sub_z, sub_size_z, sz0, sz_sub0, sz_sub1);

                float subMaxX = t_to_exit(o.x, d.x, sx_sub0, sx_sub1, step_x);
                float subMaxY = t_to_exit(o.y, d.y, sy_sub0, sy_sub1, step_y);
                float subMaxZ = t_to_exit(o.z, d.z, sz_sub0, sz_sub1, step_z);

                float t_cur = t0;
                while (sub_x >= 0 && sub_x < 4 && sub_y >= 0 && sub_y < 4 && sub_z >= 0 && sub_z < 4 && t_cur <= brick_exit_t) {
                    int bit = (sub_x << 4) | (sub_y << 2) | sub_z;
                    if (brick_mask & (1ULL << bit)) {
                        search_start_t = t_cur; // 射线进入第一个含占据子块的时刻
                        break;
                    }
                    if (subMaxX < subMaxY) {
                        if (subMaxX < subMaxZ) {
                            t_cur = subMaxX;
                            sub_x += step_x;
                            if (sub_x < 0 || sub_x >= 4) break;
                            sub_bounds(sub_x, sub_size_x, sx0, sx_sub0, sx_sub1);
                            subMaxX = t_to_exit(o.x, d.x, sx_sub0, sx_sub1, step_x);
                        } else {
                            t_cur = subMaxZ;
                            sub_z += step_z;
                            if (sub_z < 0 || sub_z >= 4) break;
                            sub_bounds(sub_z, sub_size_z, sz0, sz_sub0, sz_sub1);
                            subMaxZ = t_to_exit(o.z, d.z, sz_sub0, sz_sub1, step_z);
                        }
                    } else {
                        if (subMaxY < subMaxZ) {
                            t_cur = subMaxY;
                            sub_y += step_y;
                            if (sub_y < 0 || sub_y >= 4) break;
                            sub_bounds(sub_y, sub_size_y, sy0, sy_sub0, sy_sub1);
                            subMaxY = t_to_exit(o.y, d.y, sy_sub0, sy_sub1, step_y);
                        } else {
                            t_cur = subMaxZ;
                            sub_z += step_z;
                            if (sub_z < 0 || sub_z >= 4) break;
                            sub_bounds(sub_z, sub_size_z, sz0, sz_sub0, sz_sub1);
                            subMaxZ = t_to_exit(o.z, d.z, sz_sub0, sz_sub1, step_z);
                        }
                    }
                }
            }

            int out_idx = idx * 3;
            for (int64_t i = off0; i < off1; ++i) {
                int64_t vx = keys[i * 3 + 0];
                int64_t vy = keys[i * 3 + 1];
                int64_t vz = keys[i * 3 + 2];
                float3 vmin = make_float3(origin.x + voxel_size * vx,
                                          origin.y + voxel_size * vy,
                                          origin.z + voxel_size * vz);
                float3 vmax = make_float3(vmin.x + voxel_size,
                                          vmin.y + voxel_size,
                                          vmin.z + voxel_size);
                float vt0, vt1;
                if (!intersect_aabb(o, d, vmin, vmax, vt0, vt1) || vt1 < 0.f) {
                    continue;
                }
                if (vt0 < 0.f) vt0 = 0.f;
                // 只渲染当前 coarse brick 内且进入首个含占据子块之后的命中段
                if (vt0 + eps < search_start_t || vt0 - eps > brick_exit_t) continue;
                if (out[out_idx + 0] < 0.95f) {
                    out[out_idx + 0] += 0.1f * (1.f - out[out_idx + 0]);
                    out[out_idx + 1] += 0.1f * (1.f - out[out_idx + 1]);
                    out[out_idx + 2] += 0.1f * (1.f - out[out_idx + 2]);
                } else {
                    return;
                }
            }
        }

        // 前进到下一个 coarse brick
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                t0 = tMaxX;
                cx += step_x;
                brick_bounds(cx, brick_size.x, dims.x, origin.x, sx0, sx1);
                tMaxX = t_to_exit(o.x, d.x, sx0, sx1, step_x);
                tDeltaX = delta_t(d.x, sx0, sx1);
            } else {
                t0 = tMaxZ;
                cz += step_z;
                brick_bounds(cz, brick_size.z, dims.z, origin.z, sz0, sz1);
                tMaxZ = t_to_exit(o.z, d.z, sz0, sz1, step_z);
                tDeltaZ = delta_t(d.z, sz0, sz1);
            }
        } else {
            if (tMaxY < tMaxZ) {
                t0 = tMaxY;
                cy += step_y;
                brick_bounds(cy, brick_size.y, dims.y, origin.y, sy0, sy1);
                tMaxY = t_to_exit(o.y, d.y, sy0, sy1, step_y);
                tDeltaY = delta_t(d.y, sy0, sy1);
            } else {
                t0 = tMaxZ;
                cz += step_z;
                brick_bounds(cz, brick_size.z, dims.z, origin.z, sz0, sz1);
                tMaxZ = t_to_exit(o.z, d.z, sz0, sz1, step_z);
                tDeltaZ = delta_t(d.z, sz0, sz1);
            }
        }
    }
}

} // namespace

torch::Tensor rasterize_forward_cuda(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    int64_t height,
    int64_t width,
    float origin_x,
    float origin_y,
    float origin_z,
    float voxel_size,
    int64_t dim_x,
    int64_t dim_y,
    int64_t dim_z,
    int64_t brick_x,
    int64_t brick_y,
    int64_t brick_z,
    int64_t coarse_res) {
    // 检查并转换
    if (!rays_o.is_cuda() || !rays_d.is_cuda())
        throw std::invalid_argument("rays_o and rays_d must be CUDA tensors");
    auto ro = rays_o.contiguous().to(torch::kFloat32);
    auto rd = rays_d.contiguous().to(torch::kFloat32);
    auto offsets = coarse_offsets.contiguous();
    auto keys = voxel_keys.contiguous();
    auto mask = coarse_mask.contiguous();

    const int64_t num_rays = ro.size(0);
    auto out = torch::zeros({height, width, 3}, torch::TensorOptions().device(ro.device()).dtype(torch::kFloat32));


    // int minGridSize = 0;
    // int blockSize = 0;
    // // 动态共享内存用 0（本 kernel 没用动态 shared）
    // cudaOccupancyMaxPotentialBlockSize(
    //     &minGridSize,      // 返回最小 grid 大小
    //     &blockSize,        // 返回推荐 block size
    //     rasterize_kernel,  // kernel 函数指针
    //     0,                 // 动态 shared bytes
    //     0);                // block 大小时的上限 0=无上限

    // const int threads = blockSize;
    // const int blocks = (static_cast<int>(num_rays) + threads - 1) / threads;

    const int threads = 256;
    const int blocks = (static_cast<int>(num_rays) + threads - 1) / threads;
    rasterize_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        ro.data_ptr<float>(),
        rd.data_ptr<float>(),
        offsets.data_ptr<int64_t>(),
        keys.data_ptr<int64_t>(),
        mask.data_ptr<uint64_t>(),
        out.data_ptr<float>(),
        num_rays,
        width,
        make_float3(origin_x, origin_y, origin_z),
        voxel_size,
        make_int3(static_cast<int>(dim_x), static_cast<int>(dim_y), static_cast<int>(dim_z)),
        make_int3(static_cast<int>(brick_x), static_cast<int>(brick_y), static_cast<int>(brick_z)),
        coarse_res);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
