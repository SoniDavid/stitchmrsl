#pragma once
#include "stitching.hh"
#include "imgui.h"

#include <memory>
#include <string>

// Application state structure with stitching integration
struct ApplicationState {
    // Camera status
    bool camera_connected[3] = {true, true, true};
    float camera_fps[3] = {30.0f, 30.0f, 30.0f};
    const char* camera_names[3] = {"izquierda", "central", "derecha"};
    
    // Camera settings
    float exposure = 0.5f;
    float gain = 0.3f;
    bool auto_wb = true;
    int sync_mode = 0;
    
    // Stitching settings
    int blend_mode = 0;
    float feather_size = 0.5f;
    float ab_weight = 0.5f;
    float bc_weight = 0.5f;
    float world_scale = 1.0f;
    
    // Recording state
    bool recording = false;
    float recording_time = 0.0f;
    int output_format = 0;
    float quality = 0.8f;
    char output_path[256] = "./output/";
    
    // Display options
    bool show_overlaps = true;
    bool show_grid = false;
    int live_mode = 1; // 0 = record, 1 = live
    int resolution = 1;
    
    // Viewport display options
    bool viewport_dockable = true;           // Allow docking the viewport
    bool viewport_embedded = false;          // Embed viewport in other windows
    bool fit_to_window = true;              // Automatically fit image to window
    bool maintain_aspect = true;            // Maintain aspect ratio when resizing
    bool show_viewport_controls = true;     // Show bottom control bar
    
    // Viewport interaction
    float viewport_zoom = 1.0f;             // Current zoom level
    ImVec2 viewport_pan = ImVec2(0, 0);     // Pan offset for manual zoom mode
    
    // Viewport layout options
    enum ViewportLayout {
        SINGLE_WINDOW = 0,      // Single movable window
        TABBED_INTERFACE = 1,   // Tabbed interface with multiple views
        EMBEDDED_PANELS = 2,    // Embedded in other panels
        SPLIT_VIEW = 3          // Split view with multiple outputs
    } viewport_layout = SINGLE_WINDOW;
    
    // Performance data
    float gpu_time = 45.0f;
    float cpu_time = 12.0f;
    float fps = 28.5f;
    float memory_usage = 2.1f;
    
    // Debug status
    bool calibration_ok = true;
    bool overlap_ab_ok = true;
    bool overlap_bc_ok = true;
    int sync_offset = 2;
    
    // Window states
    bool show_camera_setup = false;
    bool show_calibration_loader = false;
    bool show_settings = false;
    bool show_about = false;
    bool show_debug_window = true;
    bool show_performance_window = true;
    bool show_stitching_debug = false;
    
    // DPI scaling
    float dpi_scale = 1.0f;
    
    // === NEW STITCHING INTEGRATION ===
    
    // Stitching pipeline instance
    std::unique_ptr<StitchingPipeline> stitching_pipeline;
    
    // Stitching initialization state
    bool stitching_initialized = false;
    bool calibration_loaded = false;
    bool test_images_loaded = false;
    
    // File paths for initialization
    char calibration_file_path[512] = "./multi_camera_calibration_extrinsics.json";
    char test_image_paths[3][512] = {
        "./imgs/izquierda.jpg",
        "./imgs/central.jpg", 
        "./imgs/derecha.jpg"
    };
    
    // World coordinate system configuration
    WorldCoordinateSystem world_config;
    
    // Blending configuration
    BlendingConfig blend_config;
    
    // Stitching output settings
    char current_output_resolution[32] = "1080p";
    bool show_stitched_output = true;
    GLuint stitched_output_texture = 0;
    
    // Advanced stitching controls
    bool auto_update_stitching = true;  // Changed to true by default
    float world_bounds_min[2] = {-4.0f, -2.5f};
    float world_bounds_max[2] = {4.0f, 2.5f};
    float z_plane_height = 0.0f;
    int tessellation_subdivisions[2] = {200, 120};
    
    // Calibration display
    bool show_camera_frustums = false;
    bool show_overlap_regions = false;
    bool show_coverage_map = false;
    
    // Initialization methods
    bool InitializeStitching() {
        std::cout << "=== INITIALIZING STITCHING PIPELINE ===" << std::endl;
        
        if (!stitching_pipeline) {
            stitching_pipeline = std::make_unique<StitchingPipeline>();
        }
        
        // Update world config from UI
        world_config.bounds_min = glm::vec2(world_bounds_min[0], world_bounds_min[1]);
        world_config.bounds_max = glm::vec2(world_bounds_max[0], world_bounds_max[1]);
        world_config.z_plane = z_plane_height;
        
        // Set output resolution based on current selection
        if (strcmp(current_output_resolution, "720p") == 0) {
            world_config.output_width = 1280;
            world_config.output_height = 720;
        } else if (strcmp(current_output_resolution, "1080p") == 0) {
            world_config.output_width = 1920;
            world_config.output_height = 1080;
        } else if (strcmp(current_output_resolution, "4K") == 0) {
            world_config.output_width = 3840;
            world_config.output_height = 2160;
        }
        
        // Update blend config from UI  
        blend_config.feather_size = feather_size;
        blend_config.blend_mode = static_cast<BlendingConfig::BlendMode>(blend_mode);
        
        stitching_pipeline->SetWorldConfig(world_config);
        stitching_pipeline->SetBlendingConfig(blend_config);
        
        // Load calibration if not already loaded
        if (!calibration_loaded) {
            if (stitching_pipeline->LoadCalibrationData(calibration_file_path)) {
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
        
        // Initialize GPU resources
        if (!stitching_pipeline->InitializeGPUResources()) {
            std::cerr << "✗ Failed to initialize GPU resources" << std::endl;
            return false;
        }
        
        // Analyze camera geometry
        stitching_pipeline->AnalyzeCameraGeometry();
        
        stitching_initialized = true;
        std::cout << "✓ Stitching pipeline initialized successfully" << std::endl;
        
        // Perform initial render
        std::cout << "=== PERFORMING INITIAL RENDER ===" << std::endl;
        UpdateStitching();
        
        return true;
    }
    
    void UpdateStitching() {
        if (!stitching_initialized || !stitching_pipeline) {
            std::cout << "UpdateStitching called but pipeline not initialized" << std::endl;
            return;
        }
        
        std::cout << "UpdateStitching: Starting render for " << current_output_resolution << std::endl;
        
        // Update configurations if they changed
        bool config_changed = false;
        
        if (world_config.bounds_min.x != world_bounds_min[0] || 
            world_config.bounds_min.y != world_bounds_min[1] ||
            world_config.bounds_max.x != world_bounds_max[0] || 
            world_config.bounds_max.y != world_bounds_max[1] ||
            world_config.z_plane != z_plane_height) {
            
            world_config.bounds_min = glm::vec2(world_bounds_min[0], world_bounds_min[1]);
            world_config.bounds_max = glm::vec2(world_bounds_max[0], world_bounds_max[1]);
            world_config.z_plane = z_plane_height;
            
            stitching_pipeline->SetWorldConfig(world_config);
            config_changed = true;
            std::cout << "UpdateStitching: World config updated" << std::endl;
        }
        
        if (blend_config.feather_size != feather_size || 
            static_cast<int>(blend_config.blend_mode) != blend_mode) {
            
            blend_config.feather_size = feather_size;
            blend_config.blend_mode = static_cast<BlendingConfig::BlendMode>(blend_mode);
            
            stitching_pipeline->SetBlendingConfig(blend_config);
            config_changed = true;
            std::cout << "UpdateStitching: Blend config updated" << std::endl;
        }
        
        if (config_changed) {
            stitching_pipeline->AnalyzeCameraGeometry();
        }
        
        // Always render the stitched output
        std::cout << "UpdateStitching: Calling RenderStitchedOutput..." << std::endl;
        stitching_pipeline->RenderStitchedOutput(current_output_resolution);
        
        // Get the output texture
        GLuint new_texture = stitching_pipeline->GetOutputTexture(current_output_resolution);
        if (new_texture != 0) {
            stitched_output_texture = new_texture;
            std::cout << "UpdateStitching: Got output texture ID " << stitched_output_texture << std::endl;
        } else {
            std::cerr << "UpdateStitching: Failed to get output texture!" << std::endl;
        }
    }
    
    void ResetStitching() {
        std::cout << "=== RESETTING STITCHING PIPELINE ===" << std::endl;
        
        stitching_initialized = false;
        calibration_loaded = false;
        test_images_loaded = false;
        stitched_output_texture = 0;
        
        if (stitching_pipeline) {
            stitching_pipeline.reset();
        }
        
        std::cout << "✓ Stitching pipeline reset complete" << std::endl;
    }
    
    // Constructor
    ApplicationState() {
        // Initialize world config with default values
        world_config.bounds_min = glm::vec2(world_bounds_min[0], world_bounds_min[1]);
        world_config.bounds_max = glm::vec2(world_bounds_max[0], world_bounds_max[1]);
        world_config.z_plane = z_plane_height;
        world_config.output_width = 1920;
        world_config.output_height = 1080;

        // Initialize new viewport fields
        viewport_zoom = 1.0f;
        viewport_pan = ImVec2(0, 0);
        fit_to_window = true;
        maintain_aspect = true;
        show_viewport_controls = true;
        viewport_dockable = true;
        viewport_embedded = false;
        
        // Initialize blend config
        blend_config.feather_size = feather_size;
        blend_config.blend_mode = static_cast<BlendingConfig::BlendMode>(blend_mode);
        
        std::cout << "ApplicationState initialized with auto_update_stitching = " 
                  << (auto_update_stitching ? "true" : "false") << std::endl;
    }
};