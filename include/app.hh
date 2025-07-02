#pragma once
#include "stitching.hh"
#include "imgui.h"

#include <memory>
#include <string>

// Application state structure with stitching integration
struct ApplicationState {
    // Camera names (for ArUco stitching)
    const char* camera_names[3] = {"izquierda", "central", "derecha"};
    
    // Simple stitching settings (ArUco-based)
    int blend_mode = 1; // 0 = Average, 1 = Feather
    float feather_size = 0.5f;
    
    // Basic status indicators
    bool calibration_ok = true;
    bool overlap_ab_ok = true;
    bool overlap_bc_ok = true;
    
    // DPI scaling
    float dpi_scale = 1.0f;
    
    // === NEW STITCHING INTEGRATION ===
    
    // Stitching pipeline instance
    std::unique_ptr<StitchingPipeline> stitching_pipeline;
    
    // Stitching initialization state
    bool stitching_initialized = false;
    bool calibration_loaded = false;
    bool test_images_loaded = false;
    
    // File paths for initialization (updated for new approach)
    char intrinsics_file_path[512] = "./intrinsic.json";
    char extrinsics_file_path[512] = "./extrinsic.json";
    char test_image_paths[3][512] = {
        "./imgs/izquierda.jpg",
        "./imgs/central.jpg", 
        "./imgs/derecha.jpg"
    };
    
    // Legacy calibration file path (for compatibility)
    char calibration_file_path[512] = "./multi_camera_calibration_extrinsics.json";
    
    // Simple output settings
    bool show_stitched_output = true;
    GLuint stitched_output_texture = 0;
    
    // ArUco stitching controls
    bool auto_update_stitching = true;
    
    // Initialization methods (simplified for ArUco stitching)
    bool InitializeStitching() {
        std::cout << "=== INITIALIZING ARUCO STITCHING PIPELINE ===" << std::endl;
        
        if (!stitching_pipeline) {
            stitching_pipeline = std::make_unique<StitchingPipeline>();
        }
        
        // Load calibration if not already loaded (using new approach)
        if (!calibration_loaded) {
            // Try to load intrinsics and extrinsics separately first
            bool intrinsics_loaded = stitching_pipeline->LoadIntrinsicsData(intrinsics_file_path);
            bool extrinsics_loaded = false;
            
            if (intrinsics_loaded) {
                extrinsics_loaded = stitching_pipeline->LoadExtrinsicsData(extrinsics_file_path);
                if (!extrinsics_loaded) {
                    std::cout << "⚠ Extrinsics not loaded - will compute on the fly" << std::endl;
                    extrinsics_loaded = true; // Allow to proceed
                }
            }
            
            // Fallback to legacy combined file if separate files failed
            if (!intrinsics_loaded) {
                std::cout << "Trying legacy calibration file..." << std::endl;
                intrinsics_loaded = stitching_pipeline->LoadCalibrationData(calibration_file_path);
            }
            
            if (intrinsics_loaded) {
                calibration_loaded = true;
                calibration_ok = true;
                std::cout << "✓ Calibration loaded successfully" << std::endl;
            } else {
                calibration_ok = false;
                std::cerr << "✗ Failed to load calibration" << std::endl;
                return false;
            }
        }
        
        // Load test images if not already loaded
        if (!test_images_loaded) {
            std::vector<std::string> image_paths = {
                test_image_paths[0],
                test_image_paths[1], 
                test_image_paths[2]
            };
            
            if (stitching_pipeline->LoadTestImages(image_paths)) {
                test_images_loaded = true;
                std::cout << "✓ Test images loaded successfully" << std::endl;
            } else {
                std::cerr << "✗ Failed to load test images" << std::endl;
                return false;
            }
        }
        
        stitching_initialized = true;
        std::cout << "✓ ArUco stitching pipeline initialized successfully" << std::endl;
        
        return true;
    }
    
    void UpdateStitching() {
        if (!stitching_initialized || !stitching_pipeline) {
            std::cout << "UpdateStitching called but pipeline not initialized" << std::endl;
            return;
        }
        
        std::cout << "UpdateStitching: Starting ArUco-based stitching..." << std::endl;
        
        // Perform pre-computed stitching
        cv::Mat panorama = stitching_pipeline->ApplyPrecomputedStitching();
        if (!panorama.empty()) {
            // Convert OpenCV Mat to OpenGL texture
            GLuint new_texture = ConvertCVMatToTexture(panorama);
            if (new_texture != 0) {
                // Clean up old texture
                if (stitched_output_texture != 0) {
                    glDeleteTextures(1, &stitched_output_texture);
                }
                stitched_output_texture = new_texture;
                std::cout << "✓ Created stitched image texture (ID " << stitched_output_texture << ")" << std::endl;
                
                // Update status indicators
                overlap_ab_ok = true;
                overlap_bc_ok = true;
            } else {
                std::cerr << "✗ Failed to create texture from panorama!" << std::endl;
            }
        } else {
            std::cerr << "✗ ArUco stitching failed to create panorama!" << std::endl;
            overlap_ab_ok = false;
            overlap_bc_ok = false;
        }
    }
    
    void ResetStitching() {
        std::cout << "=== RESETTING STITCHING PIPELINE ===" << std::endl;
        
        stitching_initialized = false;
        calibration_loaded = false;
        test_images_loaded = false;
        
        // Clean up OpenGL texture
        if (stitched_output_texture != 0) {
            glDeleteTextures(1, &stitched_output_texture);
            stitched_output_texture = 0;
        }
        
        if (stitching_pipeline) {
            stitching_pipeline.reset();
        }
        
        std::cout << "✓ Stitching pipeline reset complete" << std::endl;
    }
    
    // Helper method to convert OpenCV Mat to OpenGL texture
    GLuint ConvertCVMatToTexture(const cv::Mat& image) {
        if (image.empty()) {
            return 0;
        }
        
        GLuint texture_id;
        glGenTextures(1, &texture_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        
        // Convert BGR to RGB if needed
        cv::Mat rgb_image;
        if (image.channels() == 3) {
            cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        } else {
            rgb_image = image;
        }
        
        // Upload to GPU
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_image.cols, rgb_image.rows, 
                     0, GL_RGB, GL_UNSIGNED_BYTE, rgb_image.data);
        
        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        glBindTexture(GL_TEXTURE_2D, 0);
        
        std::cout << "Created OpenGL texture " << texture_id << " from " 
                  << rgb_image.cols << "x" << rgb_image.rows << " image" << std::endl;
        
        return texture_id;
    }
    
    // Constructor (simplified for ArUco stitching)
    ApplicationState() {
        std::cout << "ArUco Image Stitching Application initialized" << std::endl;
    }
};