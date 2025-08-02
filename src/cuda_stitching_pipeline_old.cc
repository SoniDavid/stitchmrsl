#include "cuda_stitching.hh"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <chrono>

// Include CUDA kernel declarations
extern "C" {
    void launch_rectify_fisheye_kernel(
        float* input, float* output,
        float* map1, float* map2,
        int width, int height, int channels,
        cudaStream_t stream
    );
    
    void launch_warp_perspective_kernel(
        float* input, float* output,
        float* transform_matrix,
        int input_width, int input_height,
        int output_width, int output_height, int channels,
        cudaStream_t stream
    );
    
    void launch_blend_images_kernel(
        float* img1, float* img2, float* img3,
        float* output, int width, int height, int channels,
        int blend_mode, cudaStream_t stream
    );
    
    void launch_find_content_bounds_kernel(
        float* input, int* bounds,
        int width, int height, int channels,
        float threshold, cudaStream_t stream
    );
    
    void launch_crop_image_kernel(
        float* input, float* output,
        int input_width, int input_height,
        int crop_x, int crop_y, int crop_width, int crop_height,
        int channels, cudaStream_t stream
    );
}

CUDAStitchingPipeline::CUDAStitchingPipeline() 
    : gpu_initialized_(false), blending_mode_(DEFAULT_BLENDING_MODE) {
    
    // Initialize statistics
    stats_ = {};
    
    // Initialize camera names
    cameras_[0].name = "izquierda";
    cameras_[1].name = "central";
    cameras_[2].name = "derecha";
}

CUDAStitchingPipeline::~CUDAStitchingPipeline() {
    Cleanup();
}

bool CUDAStitchingPipeline::Initialize(const cv::Size& input_size) {
    if (gpu_initialized_) {
        std::cout << "[CUDA] Already initialized" << std::endl;
        return true;
    }
    
    std::cout << "[CUDA] Initializing CUDA stitching pipeline..." << std::endl;
    
    // Check CUDA availability
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "[CUDA] No CUDA devices found" << std::endl;
        return false;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[CUDA] Using device: " << prop.name 
              << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "[CUDA] Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // Set input size
    input_size_ = input_size;
    output_size_ = cv::Size(input_size.width * 2, input_size.height); // Initial estimate
    
    // Create CUDA stream
    cudaStreamCreate(&cuda_stream_);
    
    // Allocate GPU memory
    if (!AllocateGPUMemory(input_size)) {
        std::cerr << "[CUDA] Failed to allocate GPU memory" << std::endl;
        return false;
    }
    
    gpu_initialized_ = true;
    std::cout << "[CUDA] Initialization complete" << std::endl;
    
    return true;
}

bool CUDAStitchingPipeline::LoadCalibration(const std::string& intrinsics_path, 
                                           const std::string& extrinsics_path) {
    std::cout << "[CUDA] Loading calibration data..." << std::endl;
    
    // Load intrinsics
    std::ifstream intrinsics_file(intrinsics_path);
    if (!intrinsics_file.is_open()) {
        std::cerr << "[CUDA] Failed to open intrinsics file: " << intrinsics_path << std::endl;
        return false;
    }
    
    nlohmann::json intrinsics_json;
    intrinsics_file >> intrinsics_json;
    
    auto cameras_json = intrinsics_json.value("cameras", nlohmann::json::object());
    if (cameras_json.empty()) {
        std::cerr << "[CUDA] 'cameras' key not found in intrinsics JSON" << std::endl;
        return false;
    }
    
    // Load camera calibration data
    std::vector<std::string> camera_names = {"izquierda", "central", "derecha"};
    for (int i = 0; i < 3; ++i) {
        const std::string& name = camera_names[i];
        if (!cameras_json.contains(name)) {
            std::cerr << "[CUDA] Calibration for camera '" << name << "' not found" << std::endl;
            return false;
        }
        
        auto cam_data = cameras_json[name];
        
        // Load camera matrix
        cameras_[i].camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                cameras_[i].camera_matrix.at<double>(row, col) = cam_data["K"][row][col];
            }
        }
        
        // Load distortion coefficients
        cameras_[i].distortion_coeffs.clear();
        for (const auto& dist_row : cam_data["dist"]) {
            cameras_[i].distortion_coeffs.push_back(static_cast<double>(dist_row[0]));
        }
        
        auto img_size = cam_data["image_size"];
        cameras_[i].image_size = cv::Size(img_size[0], img_size[1]);
        cameras_[i].is_fisheye = (cam_data.value("model", "fisheye") == "fisheye");
        
        std::cout << "[CUDA] Loaded calibration for: " << name << std::endl;
    }
    
    // Load extrinsics
    std::ifstream extrinsics_file(extrinsics_path);
    if (!extrinsics_file.is_open()) {
        std::cerr << "[CUDA] Failed to open extrinsics file: " << extrinsics_path << std::endl;
        return false;
    }
    
    nlohmann::json extrinsics_json;
    extrinsics_file >> extrinsics_json;
    
    auto best_transforms = extrinsics_json["best_transforms"];
    
    // Load transformation matrices
    if (best_transforms.contains("AB_similarity")) {
        transform_matrices_[0] = cv::Mat::eye(3, 3, CV_32F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                transform_matrices_[0].at<float>(i, j) = best_transforms["AB_similarity"][i][j];
            }
        }
    }
    
    if (best_transforms.contains("BC_similarity")) {
        transform_matrices_[2] = cv::Mat::eye(3, 3, CV_32F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                transform_matrices_[2].at<float>(i, j) = best_transforms["BC_similarity"][i][j];
            }
        }
    }
    
    // Central camera uses identity transform
    transform_matrices_[1] = cv::Mat::eye(3, 3, CV_32F);
    
    // Precompute rectification maps
    for (int i = 0; i < 3; ++i) {
        cv::Mat map1, map2;
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
        
        if (cameras_[i].is_fisheye) {
            cv::Mat new_K;
            cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
                cameras_[i].camera_matrix, cameras_[i].distortion_coeffs,
                cameras_[i].image_size, R, new_K, 0.4);
            
            cv::fisheye::initUndistortRectifyMap(
                cameras_[i].camera_matrix, cameras_[i].distortion_coeffs,
                R, new_K, cameras_[i].image_size, CV_32F, map1, map2);
        } else {
            cv::initUndistortRectifyMap(
                cameras_[i].camera_matrix, cameras_[i].distortion_coeffs,
                R, cameras_[i].camera_matrix, cameras_[i].image_size, 
                CV_32F, map1, map2);
        }
        
        // Upload maps to GPU
        gpu_map1_[i].upload(map1);
        gpu_map2_[i].upload(map2);
    }
    
    std::cout << "[CUDA] Calibration data loaded successfully" << std::endl;
    return true;
}

cv::Mat CUDAStitchingPipeline::ProcessFrameTriplet(const std::string& cam1_path,
                                                  const std::string& cam2_path,
                                                  const std::string& cam3_path) {
    if (!gpu_initialized_) {
        std::cerr << "[CUDA] Pipeline not initialized" << std::endl;
        return cv::Mat();
    }
    
    auto process_start = std::chrono::high_resolution_clock::now();
    
    // Load images to GPU
    if (!LoadImagesToGPU(cam1_path, cam2_path, cam3_path)) {
        std::cerr << "[CUDA] Failed to load images to GPU" << std::endl;
        return cv::Mat();
    }
    
    // Rectify images on GPU
    if (!RectifyImagesGPU()) {
        std::cerr << "[CUDA] Failed to rectify images on GPU" << std::endl;
        return cv::Mat();
    }
    
    // Stitch images on GPU
    cv::Mat result = StitchImagesGPU();
    
    // Update statistics
    auto process_end = std::chrono::high_resolution_clock::now();
    double process_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        process_end - process_start).count();
    
    stats_.frames_processed++;
    
    // Update running average
    if (stats_.frames_processed == 1) {
        stats_.average_processing_time_ms = process_time;
    } else {
        stats_.average_processing_time_ms = 
            (stats_.average_processing_time_ms * (stats_.frames_processed - 1) + process_time) 
            / stats_.frames_processed;
    }
    
    return result;
}

bool CUDAStitchingPipeline::AllocateGPUMemory(const cv::Size& input_size) {
    try {
        // Allocate memory for input frames (3 channels, float)
        for (int i = 0; i < 3; ++i) {
            gpu_frames_[i].create(input_size, CV_32FC3);
            gpu_rectified_[i].create(input_size, CV_32FC3);
            gpu_warped_[i].create(output_size_, CV_32FC3);
        }
        
        // Allocate memory for final panorama
        gpu_panorama_.create(output_size_, CV_32FC3);
        
        std::cout << "[CUDA] Allocated GPU memory for " << input_size 
                  << " input and " << output_size_ << " output" << std::endl;
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[CUDA] Failed to allocate GPU memory: " << e.what() << std::endl;
        return false;
    }
}

bool CUDAStitchingPipeline::LoadImagesToGPU(const std::string& cam1_path,
                                           const std::string& cam2_path,
                                           const std::string& cam3_path) {
    std::vector<std::string> paths = {cam1_path, cam2_path, cam3_path};
    
    for (int i = 0; i < 3; ++i) {
        cv::Mat cpu_image = cv::imread(paths[i], cv::IMREAD_COLOR);
        if (cpu_image.empty()) {
            std::cerr << "[CUDA] Failed to load image: " << paths[i] << std::endl;
            return false;
        }
        
        // Convert to float and normalize to [0,1]
        cv::Mat float_image;
        cpu_image.convertTo(float_image, CV_32F, 1.0/255.0);
        
        // Upload to GPU
        gpu_frames_[i].upload(float_image);
    }
    
    return true;
}

bool CUDAStitchingPipeline::RectifyImagesGPU() {
    for (int i = 0; i < 3; ++i) {
        // Use OpenCV's CUDA implementation for rectification
        cv::cuda::remap(gpu_frames_[i], gpu_rectified_[i], 
                       gpu_map1_[i], gpu_map2_[i], 
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, 
                       cv::Scalar(0,0,0), cuda_stream_);
    }
    
    return true;
}

cv::Mat CUDAStitchingPipeline::StitchImagesGPU() {
    // Calculate output canvas size based on transformations
    std::vector<cv::Point2f> all_corners;
    
    for (int i = 0; i < 3; ++i) {
        std::vector<cv::Point2f> corners = {
            {0, 0}, 
            {(float)input_size_.width, 0}, 
            {(float)input_size_.width, (float)input_size_.height}, 
            {0, (float)input_size_.height}
        };
        
        if (i != 1) { // Not central camera
            cv::perspectiveTransform(corners, corners, transform_matrices_[i]);
        }
        
        all_corners.insert(all_corners.end(), corners.begin(), corners.end());
    }
    
    cv::Rect bounding_box = cv::boundingRect(all_corners);
    cv::Size canvas_size(bounding_box.width, bounding_box.height);
    
    // Update output size if needed
    if (canvas_size != output_size_) {
        output_size_ = canvas_size;
        gpu_panorama_.create(output_size_, CV_32FC3);
        for (int i = 0; i < 3; ++i) {
            gpu_warped_[i].create(output_size_, CV_32FC3);
        }
    }
    
    // Create offset transformation
    cv::Mat offset_transform = cv::Mat::eye(3, 3, CV_32F);
    offset_transform.at<float>(0, 2) = -bounding_box.x;
    offset_transform.at<float>(1, 2) = -bounding_box.y;
    
    // Warp each rectified image
    for (int i = 0; i < 3; ++i) {
        cv::Mat final_transform;
        if (i == 1) { // Central camera
            final_transform = offset_transform;
        } else {
            final_transform = offset_transform * transform_matrices_[i];
        }
        
        cv::cuda::warpPerspective(gpu_rectified_[i], gpu_warped_[i], 
                                 final_transform, output_size_, 
                                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, 
                                 cv::Scalar(0,0,0), cuda_stream_);
    }
    
    // Blend images using CUDA kernel
    cv::cuda::GpuMat gpu_warped1_float, gpu_warped2_float, gpu_warped3_float;
    gpu_warped_[0].convertTo(gpu_warped1_float, CV_32F);
    gpu_warped_[1].convertTo(gpu_warped2_float, CV_32F);
    gpu_warped_[2].convertTo(gpu_warped3_float, CV_32F);
    
    // Use max blending for now (simple and effective)
    cv::cuda::max(gpu_warped1_float, gpu_warped2_float, gpu_panorama_, cuda_stream_);
    cv::cuda::max(gpu_panorama_, gpu_warped3_float, gpu_panorama_, cuda_stream_);
    
    // Auto-crop to remove black borders
    cv::Mat cpu_result;
    gpu_panorama_.download(cpu_result);
    
    // Convert back to 8-bit
    cv::Mat result_8bit;
    cpu_result.convertTo(result_8bit, CV_8U, 255.0);
    
    // Find content bounds and crop
    cv::Mat gray;
    cv::cvtColor(result_8bit, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, 1, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (!contours.empty()) {
        auto largest_contour = *std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        
        cv::Rect crop_rect = cv::boundingRect(largest_contour);
        return result_8bit(crop_rect);
    }
    
    return result_8bit;
}

void CUDAStitchingPipeline::SetBlendingMode(int mode) {
    blending_mode_ = mode;
    std::cout << "[CUDA] Blending mode set to " << mode << std::endl;
}

void CUDAStitchingPipeline::SetOutputSize(const cv::Size& size) {
    output_size_ = size;
    std::cout << "[CUDA] Output size set to " << size << std::endl;
}

void CUDAStitchingPipeline::Cleanup() {
    if (!gpu_initialized_) {
        return;
    }
    
    std::cout << "[CUDA] Cleaning up GPU resources..." << std::endl;
    
    // Release GPU memory
    for (int i = 0; i < 3; ++i) {
        gpu_frames_[i].release();
        gpu_rectified_[i].release();
        gpu_warped_[i].release();
        gpu_map1_[i].release();
        gpu_map2_[i].release();
    }
    gpu_panorama_.release();
    
    // Destroy CUDA stream
    cudaStreamDestroy(cuda_stream_);
    
    gpu_initialized_ = false;
    std::cout << "[CUDA] Cleanup complete" << std::endl;
}