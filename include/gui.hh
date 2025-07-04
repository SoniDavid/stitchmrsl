#ifndef GUI_HH
#define GUI_HH

#include <imgui.h>
#include <imgui_internal.h>
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

    //RTPS 
    char rtsp_urls[3][512] = { "", "", "" };
    bool use_rtsp_streams = false;
    cv::VideoCapture rtsp_caps[3];  // Para acceder a los streams en tiempo real

    
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

    // Manual Adjustments
    bool manual_adjustments_enabled = false;
    bool auto_update_enabled = true;
    
    // AB Transform adjustments (izquierda -> central)
    float ab_translation_x = 0.0f;
    float ab_translation_y = 0.0f;
    float ab_rotation_deg = 0.0f;
    float ab_scale_x = 1.0f;
    float ab_scale_y = 1.0f;
    
    // BC Transform adjustments (derecha -> central)
    float bc_translation_x = 0.0f;
    float bc_translation_y = 0.0f;
    float bc_rotation_deg = 0.0f;
    float bc_scale_x = 1.0f;
    float bc_scale_y = 1.0f;

    // --- METHODS ---
    void InitializeStitching();
    void CreatePanorama();
    void ResetStitching();
    void UpdatePanoramaWithAdjustments();
    void ApplyInitialAdjustments();
    bool CaptureRTSPFrames();
};

// --- IMGUI SETTINGS HANDLER FUNCTIONS ---
void* ManualAdjustments_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name);
void ManualAdjustments_ReadLine(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line);
void ManualAdjustments_WriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf);
void RegisterManualAdjustmentsHandler(ApplicationState* app_state);

// --- GUI DRAWING FUNCTIONS ---
void DrawStitchingSetupPanel(ApplicationState& state);
void DrawImageViewerPanel(ApplicationState& state);
void DrawStatusPanel(ApplicationState& state);
void DrawManualAdjustmentsPanel(ApplicationState& state);
void DrawRTSPPanel(ApplicationState& state);

#endif // GUI_HH