#include <CL/cl.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

void printDeviceInfo(const cl::Platform& platform, const cl::Device& device) {
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl << std::endl;
}

const char* kernelCode = R"kernel(
   __kernel void matrix_multiply(__global uchar* img1, __global uchar* img2, __global uchar* result, const int rows, const int cols) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows) {
        int offset = y * cols + x;
        float intensity = ((img1[offset * 3] + img1[offset * 3 + 1] + img1[offset * 3 + 2]) +
                          (img2[offset * 3] + img2[offset * 3 + 1] + img2[offset * 3 + 2])) / 6.0f;
        result[offset] = (uchar)(255 * intensity / 510);
        printf("x: %d, y: %d, offset: %d, intensity: %f, result: %d\n", x, y, offset, intensity, result[offset]);
    }
}
    )kernel";

int main() {
    cv::Mat image1 = cv::imread(R"(E:\CPP\untitled\src\anime1280x960.jpg)");
    cv::Mat image2 = cv::imread(R"(E:\CPP\untitled\src\gora1280x960.jpg)");
    if (image1.empty() || image2.empty()) {
        printf("Images loading error\n");
        return -1;
    }
    std::cout << "image 1 size: " << image1.size() << " image 2 size: " << image2.size() << "\n";

    // OpenCL setup
    cl::Platform platform;
    cl::Platform::get(&platform);

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0]);

    // Load and compile OpenCL kernel
    cl::Program program(context, kernelCode);
    cl_int err = program.build();
    if (err != CL_SUCCESS) {
        std::cerr << "Error Build OpenCL: " << err << std::endl;
        std::cerr << "Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
        return -1;
    }

    cl::Kernel kernel(program, "computeIntensity");

    // Allocate OpenCL buffers
    cl::Buffer bufferImage1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image1.total(), image1.data);
    cl::Buffer bufferImage2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image2.total(), image2.data);
    cl::Buffer bufferResult(context, CL_MEM_WRITE_ONLY, image1.total());

    // Set kernel arguments
    kernel.setArg(0, bufferImage1);
    kernel.setArg(1, bufferImage2);
    kernel.setArg(2, bufferResult);
    kernel.setArg(3, image1.rows);
    kernel.setArg(4, image1.cols);

    // Launch kernel
    const cl::NDRange globalSize(image1.cols, image1.rows);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize);

    // Read the result back to the host
    cv::Mat result(image1.size(), CV_8UC1);
    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, result.total(), result.data);
    cv::imwrite(R"(E:\CPP\untitled\res\ex.jpg)", result);
    // Save the result images
//    for (int iter = 1; iter <= 10; iter++) {
//        cv::Mat result(image1.rows, image1.cols, CV_8UC1, resultHost);
//        cv::imwrite("../res/ex" + std::to_string(iter) + ".jpg", result);
//    }
    return 0;
}