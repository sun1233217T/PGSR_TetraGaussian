#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;

    // 获取GPU设备的数量
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 2) {
        std::cerr << "This test requires at least two GPUs." << std::endl;
        return -1;
    }

    // 检查GPU0是否能访问GPU1
    int canAccess = 0;
    cudaDeviceCanAccessPeer(&canAccess, 0, 1);

    if (canAccess) {
        std::cout << "GPU0 can access GPU1's memory." << std::endl;

        // 启用GPU0访问GPU1的内存
        cudaDeviceEnablePeerAccess(1, 0);

        // 分配GPU内存
        int *d_data0, *d_data1;
        cudaSetDevice(0);
        cudaMalloc(&d_data0, sizeof(int) * 10);  // GPU0上的内存
        cudaSetDevice(1);
        cudaMalloc(&d_data1, sizeof(int) * 10);  // GPU1上的内存

        // 在GPU1上写入数据
        cudaSetDevice(1);
        int h_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        cudaMemcpy(d_data1, h_data, sizeof(int) * 10, cudaMemcpyHostToDevice);

        // 从GPU0访问GPU1的内存
        cudaSetDevice(0);
        cudaMemcpy(d_data0, d_data1, sizeof(int) * 10, cudaMemcpyDeviceToDevice);

        // 输出GPU0上的数据
        cudaMemcpy(h_data, d_data0, sizeof(int) * 10, cudaMemcpyDeviceToHost);
        std::cout << "Data on GPU0 after copying from GPU1: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << h_data[i] << " ";
        }
        std::cout << std::endl;

        // 释放资源
        cudaFree(d_data0);
        cudaFree(d_data1);

        // 禁用P2P访问
        cudaDeviceDisablePeerAccess(1);
    } else {
        std::cout << "GPU0 cannot access GPU1's memory." << std::endl;
    }

    return 0;
}
