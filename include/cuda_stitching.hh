#ifndef CUDA_STITCHING_HH
#define CUDA_STITCHING_HH

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp> 
#include <opencv2/calib3d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <string>
#include <memory>

class CUDAStitchingPipeline {
public:
    CUDAStitchingPipeline();
    ~CUDAStitchingPipeline();
    
    // Initialize GPU resources
    bool Initialize(const cv::Size& input_size);
    
    // Load camera calibration data
    bool LoadCalibration(const std::string& intrinsics_path, 
                        const std::string& extrinsics_path);
    
    // Process a synchronized frame triplet
    cv::Mat ProcessFrameTriplet(const std::string& cam1_path,
                               const std::string& cam2_path, 
                               const std::string& cam3_path);

    std::vector<cv::Mat> GetRectifiedFrames(const std::string& cam1_path,
                                           const std::string& cam2_path,
                                           const std::string& cam3_path);
    
    // Batch process multiple triplets
    std::vector<cv::Mat> ProcessBatch(const std::vector<std::tuple<std::string, std::string, std::string>>& triplets);
    
    // Set processing parameters
    void SetBlendingMode(int mode); // 0=max, 1=average, 2=feathering
    void SetOutputSize(const cv::Size& size);
    void SetFeatheringRadius(int radius); // Set feathering falloff distance
    
    // Get processing statistics
    struct ProcessingStats {
        double average_processing_time_ms;
        size_t frames_processed;
        size_t gpu_memory_used_mb;
        double gpu_utilization_percent;
    };
    ProcessingStats GetStats() const { return stats_; }
    
    // Cleanup GPU resources
    void Cleanup();

private:
    // GPU memory allocation
    bool AllocateGPUMemory(const cv::Size& input_size);
    
    // Load images to GPU
    bool LoadImagesToGPU(const std::string& cam1_path,
                        const std::string& cam2_path,
                        const std::string& cam3_path);
    
    // CUDA rectification
    bool RectifyImagesGPU();
    
    // CUDA warping and stitching
    cv::Mat StitchImagesGPU();
    
    // Generate weight masks for feathering
    void GenerateWeightMasks();
    
    // Apply feathering blend
    void ApplyFeatheringBlend();
    
    // GPU memory management
    cv::cuda::GpuMat gpu_frames_[3];        // Raw input frames
    cv::cuda::GpuMat gpu_rectified_[3];     // Rectified frames
    cv::cuda::GpuMat gpu_warped_[3];        // Warped frames
    cv::cuda::GpuMat gpu_panorama_;         // Final panorama
    
    // Weight masks for feathering
    cv::cuda::GpuMat gpu_weight_masks_[3];  // Distance-based weight masks
    cv::cuda::GpuMat gpu_temp_blend_;       // Temporary blending buffer
    
    // Rectification maps (GPU)
    cv::cuda::GpuMat gpu_map1_[3];
    cv::cuda::GpuMat gpu_map2_[3];
    
    // Transformation matrices
    cv::Mat transform_matrices_[3];
    
    // Camera calibration data
    struct CameraCalibration {
        cv::Mat camera_matrix;
        cv::Mat distortion_coeffs;
        cv::Size image_size;
        std::string name;
        bool is_fisheye = true;
    } cameras_[3];
    
    // Processing parameters
    cv::Size input_size_;
    cv::Size output_size_;
    int blending_mode_;
    int feathering_radius_;  // Feathering falloff distance in pixels
    
    // GPU resources
    bool gpu_initialized_;
    cv::cuda::Stream cuda_stream_;
    
    // Statistics
    mutable ProcessingStats stats_;
    std::chrono::high_resolution_clock::time_point last_process_start_;
    
    // Constants
    static constexpr int MAX_BATCH_SIZE = 10;
    static constexpr int DEFAULT_BLENDING_MODE = 0; // max blending
    static constexpr int DEFAULT_FEATHERING_RADIUS = 50; // pixels
};

// Using OpenCV CUDA functions only - no custom kernels needed!

#endif // CUDA_STITCHING_HH