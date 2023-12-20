#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

void printDeviceInfo(const cl::Platform& platform, const cl::Device& device) {
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl << std::endl;
}

std::vector<cl_long> getMatrix(const int size) {
    const int matrSize = size * size;
    std::vector<cl_long> matr(matrSize);
    for (int i = 0; i < matrSize; i++) {
        matr[i] = rand() % 10;
    }
    return matr;
}

void printMatrix(const std::vector<cl_long>& matr, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf(" %lld ", matr[i * size + j]);
        }
        printf("\n");
    }
}

void compareMatrix(const std::vector<cl_long>& f, const std::vector<cl_long>& s, const int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (f[i * size + j] != s[i * size + j]) {
                printf("Matrixes not equal!\n");
                return;
            }
        }
    }
    printf("Matrixes is equal!\n");
}

std::vector<cl_long> mult(const std::vector<cl_long>& A, const std::vector<cl_long>& B, const int size) {
    std::vector<cl_long> C(size * size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
    return C;
}

const char* kernelCode = R"(
    __kernel void matrix_multiply(__global long* A, __global long* B, __global long* C, const int size) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        long value = 0;
        for (int k = 0; k < size; ++k) {
            value += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = value;
    }
    )";

int main() {
    const int size = 512;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    printDeviceInfo(platform, device);
//    for (int i = 0; i < 10; i++) {
        std::vector<cl_long> A = getMatrix(size);
        std::vector<cl_long> B = getMatrix(size);
        std::vector<cl_long> C(size * size);
        auto start1 = std::chrono::system_clock::now();
        std::vector<cl_long> CPU = mult(A, B, size);
        auto end1 = std::chrono::system_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
        std::cout << "\nTime CPU: " << duration1 << " milliseconds\n";

        cl::Context context(devices);
        cl::CommandQueue queue(context, device);

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * size * size, A.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * size * size, B.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(cl_long) * size * size);

        cl::Program program(context, kernelCode);
        program.build(devices);

        cl::Kernel kernel(program, "matrix_multiply");
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, size);

        auto start = std::chrono::system_clock::now();
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size, size));
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "\nTime GPU: " << duration << " milliseconds\n";

        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(cl_long) * size * size, C.data());
        compareMatrix(C, CPU, size);
//    }

    return 0;
}
