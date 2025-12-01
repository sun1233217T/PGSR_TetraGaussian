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

// 体积近似着色：对段端点做一次三线性插值并用梯形近似密度/颜色。
__device__ inline void shade_voxel_segment(
    const float3& p0,
    const float3& p1,
    const float3& vmin,
    const float3& vmax,
    int vx,
    int vy,
    int vz,
    const float* __restrict__ vertex_sigma,
    const float* __restrict__ vertex_color,
    const uint8_t* __restrict__ vertex_valid,
    int3 v_dims,
    float& transmittance,
    float* __restrict__ out_rgb) {
    auto clamp01 = [](float v) { return fminf(fmaxf(v, 0.f), 1.f); };
    auto idx = [&](int x, int y, int z) {
        return (x * v_dims.y + y) * v_dims.z + z;
    };
    auto sample = [&](const float3& p, float& sigma_out, float3& color_out) {
        float ux = clamp01((p.x - vmin.x) / (vmax.x - vmin.x));
        float uy = clamp01((p.y - vmin.y) / (vmax.y - vmin.y));
        float uz = clamp01((p.z - vmin.z) / (vmax.z - vmin.z));
        float w[2] = {1.f - ux, ux};
        float v[2] = {1.f - uy, uy};
        float t[2] = {1.f - uz, uz};
        int ix0 = vx;
        int ix1 = min(vx + 1, v_dims.x - 1);
        int iy0 = vy;
        int iy1 = min(vy + 1, v_dims.y - 1);
        int iz0 = vz;
        int iz1 = min(vz + 1, v_dims.z - 1);
        const int xs[2] = {ix0, ix1};
        const int ys[2] = {iy0, iy1};
        const int zs[2] = {iz0, iz1};

        float sigma_acc = 0.f;
        float3 c_acc = make_float3(0.f, 0.f, 0.f);
        float weight_sum = 0.f;
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 2; ++c) {
                    float wgt = w[a] * v[b] * t[c];
                    int id = idx(xs[a], ys[b], zs[c]);
                    if (vertex_valid && vertex_valid[id] == 0) continue;
                    float sig = vertex_sigma ? vertex_sigma[id] : 0.0f;
                    int base = id * 3;
                    float cx = vertex_color ? vertex_color[base + 0] : 0.0f;
                    float cy = vertex_color ? vertex_color[base + 1] : 0.0f;
                    float cz = vertex_color ? vertex_color[base + 2] : 0.0f;
                    sigma_acc += wgt * sig;
                    c_acc.x += wgt * cx;
                    c_acc.y += wgt * cy;
                    c_acc.z += wgt * cz;
                    weight_sum += wgt;
                }
            }
        }
        if (weight_sum > 0.f) {
            float inv = 1.f / weight_sum;
            sigma_out = sigma_acc * inv;
            color_out = make_float3(c_acc.x * inv, c_acc.y * inv, c_acc.z * inv);
        } else {
            sigma_out = 0.f;
            color_out = make_float3(0.f, 0.f, 0.f);
        }
    };

    float sigma0, sigma1;
    float3 col0, col1;
    sample(p0, sigma0, col0);
    sample(p1, sigma1, col1);
    float sigma_m = 0.5f * (sigma0 + sigma1);
    float3 col_m = make_float3(0.5f * (col0.x + col1.x),
                               0.5f * (col0.y + col1.y),
                               0.5f * (col0.z + col1.z));

    float dt = sqrtf((p1.x - p0.x) * (p1.x - p0.x) +
                     (p1.y - p0.y) * (p1.y - p0.y) +
                     (p1.z - p0.z) * (p1.z - p0.z));
    if (dt <= 0.f) return;
    float alpha = 1.f - __expf(-sigma_m * dt);
    float weight = transmittance * alpha;
    out_rgb[0] += weight * col_m.x;
    out_rgb[1] += weight * col_m.y;
    out_rgb[2] += weight * col_m.z;
    transmittance *= __expf(-sigma_m * dt);
}

__global__ void rasterize_kernel(
    const float* __restrict__ rays_o,
    const float* __restrict__ rays_d,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ keys,
    const uint64_t* __restrict__ mask,
    const float* __restrict__ vertex_sigma,  // dense grid [Dx+1,Dy+1,Dz+1]
    const float* __restrict__ vertex_color,  // dense grid [Dx+1,Dy+1,Dz+1,3] packed
    const uint8_t* __restrict__ vertex_valid,// dense grid mask same shape
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
    float transmittance = 1.f;
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
            float* out_ptr = out + out_idx;
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
                float t_start = fmaxf(vt0, search_start_t);
                float t_end = fminf(vt1, brick_exit_t);
                if (t_end <= t_start) continue;
                float3 p0_seg = make_float3(o.x + d.x * t_start,
                                            o.y + d.y * t_start,
                                            o.z + d.z * t_start);
                float3 p1_seg = make_float3(o.x + d.x * t_end,
                                            o.y + d.y * t_end,
                                            o.z + d.z * t_end);
                shade_voxel_segment(
                    p0_seg, p1_seg, vmin, vmax, static_cast<int>(vx), static_cast<int>(vy), static_cast<int>(vz),
                    vertex_sigma, vertex_color, vertex_valid,
                    make_int3(dims.x + 1, dims.y + 1, dims.z + 1),
                    transmittance, out_ptr);
                if (transmittance < 1e-4f) return; // 透射率很低时提前退出
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
    const torch::Tensor& vertex_sigma,
    const torch::Tensor& vertex_color,
    const torch::Tensor& vertex_mask,
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
    const bool has_vertex = vertex_sigma.defined() && vertex_color.defined() &&
                            vertex_sigma.numel() > 0 && vertex_color.numel() > 0;
    auto v_sigma = has_vertex ? vertex_sigma.to(ro.device()).contiguous().to(torch::kFloat32) : torch::Tensor();
    auto v_color = has_vertex ? vertex_color.to(ro.device()).contiguous().to(torch::kFloat32) : torch::Tensor();
    auto v_mask  = (vertex_mask.defined() && vertex_mask.numel() > 0)
        ? vertex_mask.to(ro.device()).contiguous()
        : torch::Tensor();

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
        has_vertex ? v_sigma.data_ptr<float>() : nullptr,
        has_vertex ? v_color.data_ptr<float>() : nullptr,
        v_mask.defined() ? v_mask.data_ptr<uint8_t>() : nullptr,
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
