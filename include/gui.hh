#ifndef GUI_HH
#define GUI_HH

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include "stitching.hh"

// Helper to convert cv::Mat to an OpenGL texture for ImGui
void MatToTexture(const cv::Mat& mat, GLuint& texture_id);

// --- APPLICATION STATE ---
struct ApplicationState {
    // Stitching Pipeline
    std::unique_ptr<StitchingPipeline> pipeline;
    bool stitching_initialized = false;
    
    // File Paths
    char intrinsics_file_path[256] = "intrinsic.json";
    char extrinsics_file_path[256] = "extrinsic.json";
    char test_image_paths[3][256] = {
        "imgs/izquierda.jpg",
        "imgs/central.jpg",
        "imgs/derecha.jpg"
    };
    const char* camera_names[3] = {"Izquierda", "Central", "Derecha"};

    // Status
    bool calibration_loaded = false;
    bool test_images_loaded = false;
    std::string status_message = "Initialize pipeline to begin.";

    // Output
    cv::Mat stitched_mat;
    GLuint stitched_texture_id = 0;

    // Blending Options
    int selected_blending_mode = 1; // Default to Feathering
    const char* blending_modes[2] = {"Average", "Feathering"};

    // Viewer Options
    float rotation_degrees = 0.0f;

    // --- METHODS ---
    void InitializeStitching();
    void CreatePanorama();
    void ResetStitching();
};

// --- GUI DRAWING FUNCTIONS ---
void DrawStitchingSetupPanel(ApplicationState& state);
void DrawImageViewerPanel(ApplicationState& state);
void DrawStatusPanel(ApplicationState& state);

#endif // GUI_HH