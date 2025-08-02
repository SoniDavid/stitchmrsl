#include "cuda_stitching.hh"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <chrono>

CUDAStitchingPipeline::CUDAStitchingPipeline() 
    : blending_mode_(DEFAULT_BLENDING_MODE), feathering_radius_(DEFAULT_FEATHERING_RADIUS),
      gpu_initialized_(false) {
    
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
    int device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (device_count == 0) {
        std::cerr << "[CUDA] No CUDA devices found" << std::endl;
        return false;
    }
    
    std::cout << "[CUDA] Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Set input size
    input_size_ = input_size;
    output_size_ = cv::Size(input_size.width * 2, input_size.height); // Initial estimate
    
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
        std::vector<double> distortion_coeffs;
        for (const auto& dist_row : cam_data["dist"]) {
            distortion_coeffs.push_back(static_cast<double>(dist_row[0]));
        }
        cameras_[i].distortion_coeffs = cv::Mat(distortion_coeffs);
        
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
    
    // Load transformation matrices - Following GUI stitching logic exactly
    
    // AB transform is central -> izquierda, we need to invert it for izq -> central
    if (best_transforms.contains("AB_similarity")) {
        cv::Mat ab_transform_3x3 = cv::Mat::eye(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                ab_transform_3x3.at<double>(i, j) = best_transforms["AB_similarity"][i][j];
            }
        }
        // Invert to get izq -> central (matching GUI logic)
        cv::Mat ab_inverted;
        cv::invert(ab_transform_3x3, ab_inverted);
        ab_inverted.convertTo(transform_matrices_[0], CV_32F); // izquierda camera
        
        std::cout << "[CUDA] Loaded and inverted AB transform (izq->central)" << std::endl;
    }
    
    // BC transform is derecha -> central, use directly
    if (best_transforms.contains("BC_similarity")) {
        cv::Mat bc_transform_3x3 = cv::Mat::eye(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                bc_transform_3x3.at<double>(i, j) = best_transforms["BC_similarity"][i][j];
            }
        }
        bc_transform_3x3.convertTo(transform_matrices_[2], CV_32F); // derecha camera
        
        std::cout << "[CUDA] Loaded BC transform (derecha->central)" << std::endl;
    }
    
    // Central camera uses identity transform
    transform_matrices_[1] = cv::Mat::eye(3, 3, CV_32F);
    
    // Precompute rectification maps - EXACTLY matching GUI stitching.cc logic
    for (int i = 0; i < 3; ++i) {
        cv::Mat map1, map2;
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);  // No rectification rotation (matching GUI)
        cv::Size size = cameras_[i].image_size;
        
        // Always use fisheye model (matching GUI assumption)
        cv::Mat new_K;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
            cameras_[i].camera_matrix,
            cameras_[i].distortion_coeffs,
            size,
            R,
            new_K,
            0.4  // balance parameter (matching GUI exactly)
        );
        
        cv::fisheye::initUndistortRectifyMap(
            cameras_[i].camera_matrix,
            cameras_[i].distortion_coeffs,
            R,
            new_K,  // Use new_K for zooming out/in (matching GUI comment)
            size,
            CV_32FC1,  // Use CV_32FC1 to get separate X and Y maps for CUDA
            map1,      // X coordinates
            map2       // Y coordinates
        );
        
        std::cout << "[CUDA] Generated fisheye rectification maps for " << cameras_[i].name << std::endl;
        
        // Upload maps to GPU (already CV_32F format)
        gpu_map1_[i].upload(map1);  // X map
        gpu_map2_[i].upload(map2);  // Y map
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

std::vector<cv::Mat> CUDAStitchingPipeline::GetRectifiedFrames(const std::string& cam1_path,
                                                               const std::string& cam2_path,
                                                               const std::string& cam3_path) {
    std::vector<cv::Mat> rectified_frames(3);
    
    if (!gpu_initialized_) {
        std::cerr << "[CUDA] Pipeline not initialized" << std::endl;
        return rectified_frames;
    }
    
    // Load images to GPU
    if (!LoadImagesToGPU(cam1_path, cam2_path, cam3_path)) {
        std::cerr << "[CUDA] Failed to load images to GPU" << std::endl;
        return rectified_frames;
    }
    
    // Rectify images on GPU
    if (!RectifyImagesGPU()) {
        std::cerr << "[CUDA] Failed to rectify images on GPU" << std::endl;
        return rectified_frames;
    }
    
    // Download rectified frames to CPU
    for (int i = 0; i < 3; ++i) {
        gpu_rectified_[i].download(rectified_frames[i]);
    }
    
    return rectified_frames;
}

bool CUDAStitchingPipeline::AllocateGPUMemory(const cv::Size& input_size) {
    try {
        // Allocate memory for input frames (3 channels, float)
        for (int i = 0; i < 3; ++i) {
            gpu_frames_[i].create(input_size, CV_8UC3);
            gpu_rectified_[i].create(input_size, CV_8UC3);
            gpu_warped_[i].create(output_size_, CV_8UC3);
            gpu_weight_masks_[i].create(output_size_, CV_32F);  // Single channel float weights
        }
        
        // Allocate memory for final panorama and temporary blending
        gpu_panorama_.create(output_size_, CV_8UC3);
        gpu_temp_blend_.create(output_size_, CV_32FC3);  // Float for accumulation
        
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
        
        // Upload to GPU directly as 8UC3
        gpu_frames_[i].upload(cpu_image);
    }
    
    return true;
}

bool CUDAStitchingPipeline::RectifyImagesGPU() {
    for (int i = 0; i < 3; ++i) {
        // Use OpenCV's CUDA remap function
        cv::cuda::remap(gpu_frames_[i], gpu_rectified_[i], 
                       gpu_map1_[i], gpu_map2_[i], 
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, 
                       cv::Scalar(0,0,0));
    }
    
    return true;
}

cv::Mat CUDAStitchingPipeline::StitchImagesGPU() {
    // Calculate global canvas size - Following GUI stitching logic exactly
    
    // Get corner points for each rectified image
    std::vector<cv::Point2f> corners_izq = {
        {0,0}, {(float)input_size_.width, 0}, 
        {(float)input_size_.width, (float)input_size_.height}, {0, (float)input_size_.height}
    };
    std::vector<cv::Point2f> corners_central = {
        {0,0}, {(float)input_size_.width, 0}, 
        {(float)input_size_.width, (float)input_size_.height}, {0, (float)input_size_.height}
    };
    std::vector<cv::Point2f> corners_der = {
        {0,0}, {(float)input_size_.width, 0}, 
        {(float)input_size_.width, (float)input_size_.height}, {0, (float)input_size_.height}
    };
    
    // Transform corners using the correct transformations
    std::vector<cv::Point2f> transformed_corners_izq, transformed_corners_der;
    cv::perspectiveTransform(corners_izq, transformed_corners_izq, transform_matrices_[0]); // izq->central
    cv::perspectiveTransform(corners_der, transformed_corners_der, transform_matrices_[2]); // der->central
    
    // Combine all corners to calculate bounding box
    std::vector<cv::Point2f> all_corners;
    all_corners.insert(all_corners.end(), corners_central.begin(), corners_central.end());
    all_corners.insert(all_corners.end(), transformed_corners_izq.begin(), transformed_corners_izq.end());
    all_corners.insert(all_corners.end(), transformed_corners_der.begin(), transformed_corners_der.end());
    
    cv::Rect bounding_box = cv::boundingRect(all_corners);
    cv::Size canvas_size(bounding_box.width, bounding_box.height);
    
    std::cout << "[CUDA] Global canvas calculated: " << canvas_size << std::endl;
    
    // Update output size if needed
    if (canvas_size != output_size_) {
        output_size_ = canvas_size;
        gpu_panorama_.create(output_size_, CV_8UC3);
        gpu_temp_blend_.create(output_size_, CV_32FC3);
        for (int i = 0; i < 3; ++i) {
            gpu_warped_[i].create(output_size_, CV_8UC3);
            gpu_weight_masks_[i].create(output_size_, CV_32F);
        }
        std::cout << "[CUDA] Updated GPU memory for canvas size: " << canvas_size << std::endl;
    }
    
    // Create offset transformation to shift everything into positive coordinates
    cv::Mat offset_transform = cv::Mat::eye(3, 3, CV_32F);
    offset_transform.at<float>(0, 2) = -bounding_box.x;
    offset_transform.at<float>(1, 2) = -bounding_box.y;
    
    // Warp each rectified image using OpenCV CUDA - EXACTLY as GUI does
    
    // 1. Warp central (reference image)
    cv::cuda::warpPerspective(gpu_rectified_[1], gpu_warped_[1], 
                             offset_transform, output_size_, 
                             cv::INTER_LINEAR, cv::BORDER_CONSTANT, 
                             cv::Scalar(0,0,0));
    
    // 2. Warp izquierda
    cv::Mat izq_final_transform = offset_transform * transform_matrices_[0];
    cv::cuda::warpPerspective(gpu_rectified_[0], gpu_warped_[0], 
                             izq_final_transform, output_size_, 
                             cv::INTER_LINEAR, cv::BORDER_CONSTANT, 
                             cv::Scalar(0,0,0));
    
    // 3. Warp derecha  
    cv::Mat der_final_transform = offset_transform * transform_matrices_[2];
    cv::cuda::warpPerspective(gpu_rectified_[2], gpu_warped_[2], 
                             der_final_transform, output_size_, 
                             cv::INTER_LINEAR, cv::BORDER_CONSTANT, 
                             cv::Scalar(0,0,0));
    
    std::cout << "[CUDA] All images warped to final canvas" << std::endl;
    
    // Apply blending based on selected mode
    if (blending_mode_ == 2) { // Feathering mode
        GenerateWeightMasks();
        ApplyFeatheringBlend();
        std::cout << "[CUDA] Images blended using feathering" << std::endl;
    } else if (blending_mode_ == 1) { // Average mode
        cv::cuda::GpuMat temp_result;
        cv::cuda::add(gpu_warped_[0], gpu_warped_[1], temp_result);
        cv::cuda::add(temp_result, gpu_warped_[2], temp_result);
        cv::cuda::divide(temp_result, cv::Scalar(3, 3, 3), gpu_panorama_);
        std::cout << "[CUDA] Images blended using average" << std::endl;
    } else { // Default max mode (0)
        cv::cuda::GpuMat temp_result;
        cv::cuda::max(gpu_warped_[0], gpu_warped_[1], temp_result); // max(izq, central)
        cv::cuda::max(temp_result, gpu_warped_[2], gpu_panorama_);  // max(temp, derecha)
        std::cout << "[CUDA] Images blended using max operations" << std::endl;
    }
    
    // Download result to CPU for cropping (matching GUI logic)
    cv::Mat cpu_result;
    gpu_panorama_.download(cpu_result);
    
    // Auto-crop to remove black borders - exactly as GUI does
    cv::Mat gray;
    cv::cvtColor(cpu_result, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, 1, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (!contours.empty()) {
        double max_area = 0;
        size_t max_area_idx = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_area_idx = i;
            }
        }
        cv::Rect crop_rect = cv::boundingRect(contours[max_area_idx]);
        std::cout << "[CUDA] Cropping panorama to content" << std::endl;
        return cpu_result(crop_rect);
    }
    
    std::cout << "[CUDA] No content found to crop, returning full canvas" << std::endl;
    return cpu_result;
}

void CUDAStitchingPipeline::SetBlendingMode(int mode) {
    blending_mode_ = mode;
    std::cout << "[CUDA] Blending mode set to " << mode << std::endl;
}

void CUDAStitchingPipeline::SetFeatheringRadius(int radius) {
    feathering_radius_ = std::max(1, radius);
    std::cout << "[CUDA] Feathering radius set to " << feathering_radius_ << " pixels" << std::endl;
}

void CUDAStitchingPipeline::SetOutputSize(const cv::Size& size) {
    output_size_ = size;
    std::cout << "[CUDA] Output size set to " << size << std::endl;
}

void CUDAStitchingPipeline::GenerateWeightMasks() {
    // Generate distance-based weight masks for each warped image
    for (int i = 0; i < 3; ++i) {
        // Create binary mask from warped image (non-zero pixels)
        cv::cuda::GpuMat binary_mask;
        cv::cuda::cvtColor(gpu_warped_[i], binary_mask, cv::COLOR_BGR2GRAY);
        cv::cuda::threshold(binary_mask, binary_mask, 1, 255, cv::THRESH_BINARY);
        
        // Download to CPU for distance transform (not available in CUDA)
        cv::Mat cpu_binary_mask;
        binary_mask.download(cpu_binary_mask);
        
        // Compute distance transform on CPU
        cv::Mat cpu_dist_transform;
        cv::distanceTransform(cpu_binary_mask, cpu_dist_transform, cv::DIST_L2, 3);
        
        // Create feathering weights: min(1.0, distance / feathering_radius)
        cv::Mat cpu_feather_weights;
        cpu_dist_transform.convertTo(cpu_feather_weights, CV_32F, 1.0/feathering_radius_);
        cv::min(cpu_feather_weights, 1.0, cpu_feather_weights);
        
        // Convert binary mask to float for multiplication
        cv::Mat cpu_binary_mask_f;
        cpu_binary_mask.convertTo(cpu_binary_mask_f, CV_32F, 1.0/255.0);
        
        // Multiply by binary mask to ensure zero weights outside image
        cv::Mat cpu_final_weights;
        cv::multiply(cpu_feather_weights, cpu_binary_mask_f, cpu_final_weights);
        
        // Upload final weights back to GPU
        gpu_weight_masks_[i].upload(cpu_final_weights);
    }
    
    std::cout << "[CUDA] Generated feathering weight masks with radius " << feathering_radius_ << std::endl;
}

void CUDAStitchingPipeline::ApplyFeatheringBlend() {
    // Initialize accumulation buffers
    cv::cuda::GpuMat accumulated_image(output_size_, CV_32FC3, cv::Scalar(0,0,0));
    cv::cuda::GpuMat accumulated_weights(output_size_, CV_32F, cv::Scalar(0));
    
    // Accumulate weighted contributions from each camera
    for (int i = 0; i < 3; ++i) {
        // Convert warped image to float
        cv::cuda::GpuMat warped_float;
        gpu_warped_[i].convertTo(warped_float, CV_32FC3);
        
        // Create 3-channel weight mask
        cv::cuda::GpuMat weight_3ch;
        cv::cuda::cvtColor(gpu_weight_masks_[i], weight_3ch, cv::COLOR_GRAY2BGR);
        
        // Weighted image contribution
        cv::cuda::GpuMat weighted_contribution;
        cv::cuda::multiply(warped_float, weight_3ch, weighted_contribution);
        
        // Accumulate weighted image
        cv::cuda::add(accumulated_image, weighted_contribution, accumulated_image);
        
        // Accumulate weights
        cv::cuda::add(accumulated_weights, gpu_weight_masks_[i], accumulated_weights);
    }
    
    // Normalize by accumulated weights (avoid division by zero)
    cv::cuda::GpuMat weight_3ch_norm;
    cv::cuda::cvtColor(accumulated_weights, weight_3ch_norm, cv::COLOR_GRAY2BGR);
    
    // Set minimum weight to avoid division by zero
    cv::cuda::max(weight_3ch_norm, cv::Scalar(1e-6, 1e-6, 1e-6), weight_3ch_norm);
    
    // Normalize and convert back to 8-bit
    cv::cuda::divide(accumulated_image, weight_3ch_norm, gpu_temp_blend_);
    gpu_temp_blend_.convertTo(gpu_panorama_, CV_8UC3);
    
    std::cout << "[CUDA] Applied feathering blend with normalization" << std::endl;
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
        gpu_weight_masks_[i].release();
        gpu_map1_[i].release();
        gpu_map2_[i].release();
    }
    gpu_panorama_.release();
    gpu_temp_blend_.release();
    
    gpu_initialized_ = false;
    std::cout << "[CUDA] Cleanup complete" << std::endl;
}

std::vector<cv::Mat> CUDAStitchingPipeline::ProcessBatch(const std::vector<std::tuple<std::string, std::string, std::string>>& triplets) {
    std::vector<cv::Mat> results;
    results.reserve(triplets.size());
    
    for (const auto& triplet : triplets) {
        cv::Mat result = ProcessFrameTriplet(std::get<0>(triplet), std::get<1>(triplet), std::get<2>(triplet));
        if (!result.empty()) {
            results.push_back(result);
        }
    }
    
    return results;
}

