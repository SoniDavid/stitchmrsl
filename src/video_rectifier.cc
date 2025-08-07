#include "cuda_stitching.hh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <csignal>
#include <atomic>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

// Global variables for signal handling
std::atomic<bool> g_stop_requested{false};

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n[INFO] Ctrl+C received. Stopping processing..." << std::endl;
        g_stop_requested = true;
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input_video> <camera_name> [output_video]" << std::endl;
    std::cout << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  input_video   - Path to input MP4 video file" << std::endl;
    std::cout << "  camera_name   - Camera identifier: izquierda, central, or derecha" << std::endl;
    std::cout << "  output_video  - Output rectified video path (default: <input>_rectified.mp4)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " camera1.mp4 izquierda" << std::endl;
    std::cout << "  " << program_name << " camera2.mp4 central output_rectified.mp4" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: Requires intrinsic.json in current directory for camera calibration" << std::endl;
}

class VideoRectifier {
private:
    std::string camera_name_;
    int camera_index_;
    cv::cuda::GpuMat gpu_input_frame_;
    cv::cuda::GpuMat gpu_rectified_frame_;
    cv::cuda::GpuMat gpu_map1_;
    cv::cuda::GpuMat gpu_map2_;
    bool gpu_initialized_;
    
    struct CameraData {
        std::string name;
        cv::Mat camera_matrix;
        cv::Mat distortion_coeffs;
        cv::Size image_size;
        bool is_fisheye;
    } camera_;

public:
    VideoRectifier() : camera_index_(-1), gpu_initialized_(false) {}
    
    ~VideoRectifier() {
        Cleanup();
    }

    bool Initialize(const std::string& camera_name) {
        camera_name_ = camera_name;
        
        // Map camera name to index
        if (camera_name == "izquierda") {
            camera_index_ = 0;
        } else if (camera_name == "central") {
            camera_index_ = 1;
        } else if (camera_name == "derecha") {
            camera_index_ = 2;
        } else {
            std::cerr << "[ERROR] Invalid camera name: " << camera_name << std::endl;
            std::cerr << "[ERROR] Valid names: izquierda, central, derecha" << std::endl;
            return false;
        }
        
        std::cout << "[INFO] Initializing video rectifier for camera: " << camera_name << std::endl;
        
        // Check CUDA availability
        int device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (device_count == 0) {
            std::cerr << "[ERROR] No CUDA devices found" << std::endl;
            return false;
        }
        
        std::cout << "[INFO] Found " << device_count << " CUDA device(s)" << std::endl;
        
        // Load calibration data
        if (!LoadCalibration()) {
            std::cerr << "[ERROR] Failed to load calibration data" << std::endl;
            return false;
        }
        
        gpu_initialized_ = true;
        std::cout << "[INFO] Video rectifier initialized successfully" << std::endl;
        
        return true;
    }
    
    bool ProcessVideo(const std::string& input_path, const std::string& output_path) {
        if (!gpu_initialized_) {
            std::cerr << "[ERROR] Rectifier not initialized" << std::endl;
            return false;
        }
        
        // Open input video
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) {
            std::cerr << "[ERROR] Failed to open input video: " << input_path << std::endl;
            return false;
        }
        
        // Get video properties
        int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        
        std::cout << "[INFO] Input video properties:" << std::endl;
        std::cout << "[INFO]   Frames: " << frame_count << std::endl;
        std::cout << "[INFO]   FPS: " << fps << std::endl;
        std::cout << "[INFO]   Size: " << width << "x" << height << std::endl;
        
        // Allocate GPU memory for this video size
        cv::Size frame_size(width, height);
        if (!AllocateGPUMemory(frame_size)) {
            std::cerr << "[ERROR] Failed to allocate GPU memory" << std::endl;
            return false;
        }
        
        // Initialize video writer
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::VideoWriter writer(output_path, fourcc, fps, frame_size);
        if (!writer.isOpened()) {
            std::cerr << "[ERROR] Failed to open output video writer: " << output_path << std::endl;
            return false;
        }
        
        std::cout << "[INFO] Starting video processing..." << std::endl;
        std::cout << "[INFO] Output video: " << output_path << std::endl;
        
        // Process frames
        cv::Mat cpu_frame, cpu_rectified;
        int processed_frames = 0;
        auto process_start = std::chrono::steady_clock::now();
        
        while (cap.read(cpu_frame) && !g_stop_requested) {
            auto frame_start = std::chrono::steady_clock::now();
            
            // Upload frame to GPU
            gpu_input_frame_.upload(cpu_frame);
            
            // Rectify on GPU
            cv::cuda::remap(gpu_input_frame_, gpu_rectified_frame_, 
                           gpu_map1_, gpu_map2_, 
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, 
                           cv::Scalar(0,0,0));
            
            // Download rectified frame
            gpu_rectified_frame_.download(cpu_rectified);
            
            // Write to output video
            writer.write(cpu_rectified);
            
            processed_frames++;
            
            // Progress reporting
            if (processed_frames % 30 == 0 || processed_frames == frame_count) {
                auto frame_end = std::chrono::steady_clock::now();
                double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    frame_end - frame_start).count();
                
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    frame_end - process_start).count();
                double progress = (double)processed_frames / frame_count * 100.0;
                double processing_fps = processed_frames / std::max(1.0, (double)elapsed);
                
                std::cout << "[INFO] Progress: " << std::fixed << std::setprecision(1) 
                          << progress << "% (" << processed_frames << "/" << frame_count 
                          << ") - Processing FPS: " << std::setprecision(1) << processing_fps
                          << " - Frame time: " << std::setprecision(0) << frame_time << "ms" << std::endl;
            }
        }
        
        // Cleanup
        cap.release();
        writer.release();
        
        auto process_end = std::chrono::steady_clock::now();
        double total_time = std::chrono::duration_cast<std::chrono::seconds>(
            process_end - process_start).count();
        double avg_fps = processed_frames / std::max(1.0, total_time);
        
        if (g_stop_requested) {
            std::cout << "\n[INFO] Processing interrupted by user" << std::endl;
        } else {
            std::cout << "\n[INFO] ✅ Video rectification completed successfully!" << std::endl;
        }
        
        std::cout << "[INFO] Processed frames: " << processed_frames << "/" << frame_count << std::endl;
        std::cout << "[INFO] Total processing time: " << std::fixed << std::setprecision(1) << total_time << "s" << std::endl;
        std::cout << "[INFO] Average processing FPS: " << std::setprecision(1) << avg_fps << std::endl;
        std::cout << "[INFO] Output saved to: " << output_path << std::endl;
        
        return !g_stop_requested || processed_frames > 0;
    }

private:
    bool LoadCalibration() {
        std::cout << "[INFO] Loading calibration data..." << std::endl;
        
        // Load intrinsics
        std::ifstream intrinsics_file("intrinsic.json");
        if (!intrinsics_file.is_open()) {
            std::cerr << "[ERROR] Failed to open intrinsic.json" << std::endl;
            return false;
        }
        
        nlohmann::json intrinsics_json;
        intrinsics_file >> intrinsics_json;
        
        auto cameras_json = intrinsics_json.value("cameras", nlohmann::json::object());
        if (cameras_json.empty()) {
            std::cerr << "[ERROR] 'cameras' key not found in intrinsics JSON" << std::endl;
            return false;
        }
        
        // Load specific camera calibration data
        if (!cameras_json.contains(camera_name_)) {
            std::cerr << "[ERROR] Calibration for camera '" << camera_name_ << "' not found" << std::endl;
            return false;
        }
        
        auto cam_data = cameras_json[camera_name_];
        
        // Load camera matrix
        camera_.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                camera_.camera_matrix.at<double>(row, col) = cam_data["K"][row][col];
            }
        }
        
        // Load distortion coefficients
        std::vector<double> distortion_coeffs;
        for (const auto& dist_row : cam_data["dist"]) {
            distortion_coeffs.push_back(static_cast<double>(dist_row[0]));
        }
        camera_.distortion_coeffs = cv::Mat(distortion_coeffs);
        
        auto img_size = cam_data["image_size"];
        camera_.image_size = cv::Size(img_size[0], img_size[1]);
        camera_.is_fisheye = (cam_data.value("model", "fisheye") == "fisheye");
        camera_.name = camera_name_;
        
        std::cout << "[INFO] Loaded calibration for: " << camera_name_ << std::endl;
        
        // Generate rectification maps
        GenerateRectificationMaps();
        
        return true;
    }
    
    void GenerateRectificationMaps() {
        cv::Mat map1, map2;
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);  // No rectification rotation
        cv::Size size = camera_.image_size;
        
        // Always use fisheye model (matching GUI assumption)
        cv::Mat new_K;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
            camera_.camera_matrix,
            camera_.distortion_coeffs,
            size,
            R,
            new_K,
            0.4  // balance parameter (matching GUI exactly)
        );
        
        cv::fisheye::initUndistortRectifyMap(
            camera_.camera_matrix,
            camera_.distortion_coeffs,
            R,
            new_K,  // Use new_K for zooming out/in
            size,
            CV_32FC1,  // Use CV_32FC1 to get separate X and Y maps for CUDA
            map1,      // X coordinates
            map2       // Y coordinates
        );
        
        std::cout << "[INFO] Generated fisheye rectification maps for " << camera_.name << std::endl;
        
        // Upload maps to GPU (already CV_32F format)
        gpu_map1_.upload(map1);  // X map
        gpu_map2_.upload(map2);  // Y map
    }
    
    bool AllocateGPUMemory(const cv::Size& frame_size) {
        try {
            // Allocate memory for input and rectified frames
            gpu_input_frame_.create(frame_size, CV_8UC3);
            gpu_rectified_frame_.create(frame_size, CV_8UC3);
            
            std::cout << "[INFO] Allocated GPU memory for " << frame_size << " frames" << std::endl;
            
            return true;
        } catch (const cv::Exception& e) {
            std::cerr << "[ERROR] Failed to allocate GPU memory: " << e.what() << std::endl;
            return false;
        }
    }
    
    void Cleanup() {
        if (!gpu_initialized_) {
            return;
        }
        
        std::cout << "[INFO] Cleaning up GPU resources..." << std::endl;
        
        // Release GPU memory
        gpu_input_frame_.release();
        gpu_rectified_frame_.release();
        gpu_map1_.release();
        gpu_map2_.release();
        
        gpu_initialized_ = false;
        std::cout << "[INFO] Cleanup complete" << std::endl;
    }
};

std::string generate_output_path(const std::string& input_path, const std::string& camera_name) {
    fs::path input_file(input_path);
    std::string stem = input_file.stem().string();
    std::string extension = input_file.extension().string();
    std::string parent = input_file.parent_path().string();
    
    std::string output_filename = stem + "_" + camera_name + "_rectified" + extension;
    
    if (parent.empty()) {
        return output_filename;
    } else {
        return (fs::path(parent) / output_filename).string();
    }
}

int main(int argc, char** argv) {
    // Install signal handler
    std::signal(SIGINT, signal_handler);
    
    std::cout << "[INFO] Video Rectifier v1.0 - Single Camera Video Processing" << std::endl;
    
    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }
    
    std::string input_video = argv[1];
    std::string camera_name = argv[2];
    std::string output_video;
    
    if (argc > 3) {
        output_video = argv[3];
    } else {
        output_video = generate_output_path(input_video, camera_name);
    }
    
    // Validate input file exists
    if (!fs::exists(input_video)) {
        std::cerr << "[ERROR] Input video file does not exist: " << input_video << std::endl;
        return -1;
    }
    
    // Validate calibration file exists
    if (!fs::exists("intrinsic.json")) {
        std::cerr << "[ERROR] Calibration file 'intrinsic.json' not found in current directory" << std::endl;
        return -1;
    }
    
    std::cout << "[INFO] Input video: " << input_video << std::endl;
    std::cout << "[INFO] Camera: " << camera_name << std::endl;
    std::cout << "[INFO] Output video: " << output_video << std::endl;
    
    // Initialize rectifier
    VideoRectifier rectifier;
    if (!rectifier.Initialize(camera_name)) {
        std::cerr << "[ERROR] Failed to initialize video rectifier" << std::endl;
        return -1;
    }
    
    // Process video
    if (!rectifier.ProcessVideo(input_video, output_video)) {
        std::cerr << "[ERROR] Failed to process video" << std::endl;
        return -1;
    }
    
    return 0;
}