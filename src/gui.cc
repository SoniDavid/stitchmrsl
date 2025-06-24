// Additional GUI functions for stitching integration
// Add these to gui.cc

#include "gui.hh"

// Helper function to get current time in seconds
float GetTime() {
    static auto start = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float>(now - start).count();
}

// New stitching initialization panel
void DrawStitchingInitPanel(ApplicationState& state) {
    ImGui::Begin("Stitching Initialization");
    
    // Calibration file selection
    ImGui::Text("Calibration File:");
    ImGui::InputText("##calibration_path", state.calibration_file_path, sizeof(state.calibration_file_path));
    ImGui::SameLine();
    if (ImGui::Button("Browse##cal")) {
        // TODO: Implement file dialog
    }
    
    // Test image paths
    ImGui::Text("Test Images:");
    for (int i = 0; i < 3; i++) {
        ImGui::PushID(i);
        ImGui::Text("%s:", state.camera_names[i]);
        ImGui::InputText("##image_path", state.test_image_paths[i], sizeof(state.test_image_paths[i]));
        ImGui::SameLine();
        if (ImGui::Button("Browse##img")) {
            // TODO: Implement file dialog
        }
        ImGui::PopID();
    }
    
    ImGui::Separator();
    
    // Status indicators
    ImGui::Text("Status:");
    ImVec4 green = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
    ImVec4 red = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    
    ImGui::TextColored(state.calibration_loaded ? green : red, 
                       "Calibration: %s", state.calibration_loaded ? "Loaded" : "Not Loaded");
    ImGui::TextColored(state.test_images_loaded ? green : red,
                       "Test Images: %s", state.test_images_loaded ? "Loaded" : "Not Loaded");
    ImGui::TextColored(state.stitching_initialized ? green : red,
                       "Stitching: %s", state.stitching_initialized ? "Initialized" : "Not Initialized");
    
    ImGui::Separator();
    
    // World coordinate system configuration
    if (ImGui::CollapsingHeader("World Coordinate System")) {
        ImGui::SliderFloat2("Bounds Min (m)", state.world_bounds_min, -10.0f, 0.0f);
        ImGui::SliderFloat2("Bounds Max (m)", state.world_bounds_max, 0.0f, 10.0f);
        ImGui::SliderFloat("Z-Plane Height (m)", &state.z_plane_height, -1.0f, 1.0f);
        
        ImGui::Text("Output Resolution:");
        const char* resolutions[] = {"720p", "1080p", "4K"};
        int current_res = 1; // Default to 1080p
        if (strcmp(state.current_output_resolution, "720p") == 0) current_res = 0;
        else if (strcmp(state.current_output_resolution, "4K") == 0) current_res = 2;
        
        if (ImGui::Combo("##resolution", &current_res, resolutions, IM_ARRAYSIZE(resolutions))) {
            strcpy(state.current_output_resolution, resolutions[current_res]);
        }
        
        ImGui::SliderInt2("Tessellation", state.tessellation_subdivisions, 50, 500);
    }
    
    // Blending configuration
    if (ImGui::CollapsingHeader("Blending Configuration")) {
        const char* blend_modes[] = {"Linear", "Multiband", "Feather", "None"};
        ImGui::Combo("Blend Mode", &state.blend_mode, blend_modes, IM_ARRAYSIZE(blend_modes));
        ImGui::SliderFloat("Feather Size", &state.feather_size, 0.0f, 2.0f);
    }
    
    ImGui::Separator();
    
    // Initialize button
    if (!state.stitching_initialized) {
        if (ImGui::Button("Initialize Stitching", ImVec2(-1, 0))) {
            state.InitializeStitching();
        }
    } else {
        ImGui::TextColored(green, "Stitching Ready!");
        
        if (ImGui::Button("Reset Stitching", ImVec2(-1, 0))) {
            state.ResetStitching();
        }
        
        ImGui::Checkbox("Auto Update", &state.auto_update_stitching);
        
        if (!state.auto_update_stitching) {
            if (ImGui::Button("Manual Update", ImVec2(-1, 0))) {
                std::cout << "Manual update triggered" << std::endl;
                state.UpdateStitching();
            }
        }
        
        // Force an initial render when just initialized
        static bool initial_render_done = false;
        if (state.stitching_initialized && !initial_render_done) {
            std::cout << "Performing initial stitching render..." << std::endl;
            state.UpdateStitching();
            initial_render_done = true;
        }
    }
    
    ImGui::End();
}

// Enhanced stitching panel with 3D reprojection controls
void DrawAdvancedStitchingPanel(ApplicationState& state) {
    ImGui::Begin("3D Reprojection Stitching");
    
    if (!state.stitching_initialized) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Stitching not initialized!");
        ImGui::Text("Go to 'Stitching Initialization' panel first.");
        ImGui::End();
        return;
    }
    
    // Real-time controls
    if (ImGui::CollapsingHeader("Real-time Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool config_changed = false;
        
        // Blending parameters
        if (ImGui::SliderFloat("Feather Size", &state.feather_size, 0.0f, 1.0f)) {
            config_changed = true;
        }
        
        if (ImGui::SliderFloat("A-B Weight", &state.ab_weight, 0.0f, 1.0f)) {
            config_changed = true;
        }
        
        if (ImGui::SliderFloat("B-C Weight", &state.bc_weight, 0.0f, 1.0f)) {
            config_changed = true;
        }
        
        const char* blend_modes[] = {"Linear", "Multiband", "Feather", "None"};
        if (ImGui::Combo("Blend Mode", &state.blend_mode, blend_modes, IM_ARRAYSIZE(blend_modes))) {
            config_changed = true;
        }
        
        // World coordinate adjustments
        if (ImGui::SliderFloat2("World Min", state.world_bounds_min, -10.0f, 0.0f)) {
            config_changed = true;
        }
        
        if (ImGui::SliderFloat2("World Max", state.world_bounds_max, 0.0f, 10.0f)) {
            config_changed = true;
        }
        
        if (ImGui::SliderFloat("Z-Plane", &state.z_plane_height, -1.0f, 1.0f)) {
            config_changed = true;
        }
        
        if (config_changed && state.auto_update_stitching) {
            std::cout << "Configuration changed, updating stitching..." << std::endl;
            state.UpdateStitching();
        }
        
        // Manual render button
        if (ImGui::Button("Force Render Now")) {
            std::cout << "Force render triggered" << std::endl;
            state.UpdateStitching();
        }
    }
    
    // Visualization options
    if (ImGui::CollapsingHeader("Visualization")) {
        ImGui::Checkbox("Show Camera Frustums", &state.show_camera_frustums);
        ImGui::Checkbox("Show Overlap Regions", &state.show_overlap_regions);
        ImGui::Checkbox("Show Coverage Map", &state.show_coverage_map);
        ImGui::Checkbox("Show Overlaps", &state.show_overlaps);
        ImGui::Checkbox("Show Grid", &state.show_grid);
    }
    
    // Performance metrics
    if (ImGui::CollapsingHeader("Performance")) {
        if (state.stitching_pipeline) {
            ImGui::Text("Cameras: %zu", state.stitching_pipeline->GetCameras().size());
            ImGui::Text("Resolution: %s", state.current_output_resolution);
            
            // Get world config info
            const auto& world_config = state.stitching_pipeline->GetWorldConfig();
            ImGui::Text("World Size: %.1fm x %.1fm", 
                       world_config.bounds_max.x - world_config.bounds_min.x,
                       world_config.bounds_max.y - world_config.bounds_min.y);
            ImGui::Text("Pixel Ratio: %.1f px/m", world_config.pixel_to_meter_ratio);
        }
        
        ImGui::Text("GPU Time: %.1fms", state.gpu_time);
        ImGui::Text("CPU Time: %.1fms", state.cpu_time);
        ImGui::Text("Stitching FPS: %.1f", state.fps);
        
        // Debug info about textures
        ImGui::Separator();
        ImGui::Text("Debug Info:");
        ImGui::Text("Output Texture ID: %u", state.stitched_output_texture);
        if (state.stitching_pipeline) {
            GLuint tex = state.stitching_pipeline->GetOutputTexture(state.current_output_resolution);
            ImGui::Text("Pipeline Texture ID: %u", tex);
        }
    }
    
    if (ImGui::CollapsingHeader("Cylindrical Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool config_changed = false;
        
        // Projection type selection
        const char* projection_types[] = {"Planar", "Cylindrical", "Spherical (Future)"};
        int current_projection = 1; // Default to cylindrical
        if (ImGui::Combo("Projection Type", &current_projection, projection_types, 2)) { // Only show first 2 options
            config_changed = true;
            std::cout << "Projection type changed to: " << projection_types[current_projection] << std::endl;
        }
        
        if (current_projection == 1) { // Cylindrical projection
            // Cylinder radius controls
            if (state.stitching_pipeline) {
                float current_radius = state.stitching_pipeline->GetCylinderRadius();
                
                bool auto_radius = true; // You might want to store this in ApplicationState
                if (ImGui::Checkbox("Auto Cylinder Radius", &auto_radius)) {
                    if (auto_radius) {
                        // Reset to auto-calculated radius
                        state.stitching_pipeline->SetCylinderRadius(current_radius);
                    }
                    config_changed = true;
                }
                
                if (!auto_radius) {
                    if (ImGui::SliderFloat("Cylinder Radius", &current_radius, 1.0f, 10.0f, "%.1f m")) {
                        state.stitching_pipeline->SetCylinderRadius(current_radius);
                        config_changed = true;
                    }
                } else {
                    ImGui::Text("Auto Radius: %.1f m", current_radius);
                }
                
                // Cylinder visualization info
                ImGui::Separator();
                ImGui::Text("Cylindrical Surface Info:");
                float world_width = state.world_bounds_max[0] - state.world_bounds_min[0];
                float circumference = 2.0f * M_PI * current_radius;
                float angular_coverage = world_width / current_radius; // radians
                float angular_degrees = angular_coverage * 180.0f / M_PI;
                
                ImGui::Text("  World Width: %.1f m", world_width);
                ImGui::Text("  Cylinder Circumference: %.1f m", circumference);
                ImGui::Text("  Angular Coverage: %.1f° (%.2f rad)", angular_degrees, angular_coverage);
                
                if (angular_degrees > 180.0f) {
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), 
                        "Warning: Angular coverage > 180° may cause issues");
                }
            }
            
            // Distortion mitigation info
            ImGui::Separator();
            ImGui::Text("Distortion Mitigation:");
            ImGui::TextWrapped("Cylindrical projection reduces barrel distortion artifacts "
                            "from oblique camera angles, especially for the 'izquierda' camera.");
            
            if (ImGui::Button("Optimize for Current Scene")) {
                // TODO: Implement scene-based optimization
                std::cout << "Scene optimization requested" << std::endl;
            }
        }
        
        if (config_changed && state.auto_update_stitching) {
            std::cout << "Cylindrical projection configuration changed, updating stitching..." << std::endl;
            state.UpdateStitching();
        }
    }

    // === DISTORTION ANALYSIS ===
    if (ImGui::CollapsingHeader("Distortion Analysis")) {
        if (state.stitching_pipeline) {
            const auto& cameras = state.stitching_pipeline->GetCameras();
            
            ImGui::Text("Camera Distortion Coefficients:");
            for (size_t i = 0; i < cameras.size(); i++) {
                const auto& cam = cameras[i];
                const auto& dist_coeffs = cam.intrinsics.distortion_coeffs;
                
                ImGui::PushID(i);
                if (ImGui::TreeNode(cam.name.c_str())) {
                    ImGui::Text("k1 (barrel): %.4f", dist_coeffs.size() > 0 ? dist_coeffs[0] : 0.0f);
                    ImGui::Text("k2 (barrel): %.4f", dist_coeffs.size() > 1 ? dist_coeffs[1] : 0.0f);
                    ImGui::Text("p1 (tangential): %.4f", dist_coeffs.size() > 2 ? dist_coeffs[2] : 0.0f);
                    ImGui::Text("p2 (tangential): %.4f", dist_coeffs.size() > 3 ? dist_coeffs[3] : 0.0f);
                    
                    // Highlight problematic distortion
                    if (dist_coeffs.size() > 0 && abs(dist_coeffs[0]) > 0.1f) {
                        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 
                            "High barrel distortion detected!");
                        ImGui::TextWrapped("Cylindrical projection recommended for this camera.");
                    }
                    
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }
    }

    // === PROJECTION COMPARISON ===
    if (ImGui::CollapsingHeader("Projection Comparison")) {
        ImGui::TextWrapped("Planar vs Cylindrical Projection:");
        
        if (ImGui::BeginTable("ProjectionComparison", 3, ImGuiTableFlags_Borders)) {
            ImGui::TableSetupColumn("Aspect");
            ImGui::TableSetupColumn("Planar");
            ImGui::TableSetupColumn("Cylindrical");
            ImGui::TableHeadersRow();
            
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Distortion Handling");
            ImGui::TableSetColumnIndex(1); ImGui::Text("Poor for oblique cameras");
            ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Excellent");
            
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Processing Speed");
            ImGui::TableSetColumnIndex(1); ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Fast");
            ImGui::TableSetColumnIndex(2); ImGui::Text("Moderate");
            
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Memory Usage");
            ImGui::TableSetColumnIndex(1); ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Low");
            ImGui::TableSetColumnIndex(2); ImGui::Text("Moderate");
            
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Visual Quality");
            ImGui::TableSetColumnIndex(1); ImGui::Text("Variable");
            ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Consistent");
            
            ImGui::EndTable();
        }
        
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), 
            "Recommendation: Use Cylindrical projection for your multi-camera setup.");
    }

    ImGui::End();
}

// Helper function to calculate image display transform
void CalculateImageDisplayTransform(const ImVec2& canvas_size, const ImVec2& canvas_pos, 
                                   ApplicationState& state, ImVec2& out_image_size, ImVec2& out_image_pos) {
    // Get actual texture dimensions based on resolution
    int tex_width = 1920, tex_height = 1080; // Default 1080p
    if (strcmp(state.current_output_resolution, "720p") == 0) {
        tex_width = 1280; tex_height = 720;
    } else if (strcmp(state.current_output_resolution, "4K") == 0) {
        tex_width = 3840; tex_height = 2160;
    }
    
    float aspect_ratio = (float)tex_width / tex_height;
    
    if (state.fit_to_window) {
        // Fit to window while maintaining aspect ratio
        float canvas_aspect = canvas_size.x / canvas_size.y;
        
        if (aspect_ratio > canvas_aspect) {
            // Fit to width
            out_image_size.x = canvas_size.x;
            out_image_size.y = canvas_size.x / aspect_ratio;
        } else {
            // Fit to height
            out_image_size.y = canvas_size.y;
            out_image_size.x = canvas_size.y * aspect_ratio;
        }
        
        // Center the image
        out_image_pos.x = canvas_pos.x + (canvas_size.x - out_image_size.x) * 0.5f;
        out_image_pos.y = canvas_pos.y + (canvas_size.y - out_image_size.y) * 0.5f;
        
    } else {
        // Manual zoom and pan
        out_image_size.x = tex_width * state.viewport_zoom * state.dpi_scale;
        out_image_size.y = tex_height * state.viewport_zoom * state.dpi_scale;
        
        // Apply pan offset
        out_image_pos.x = canvas_pos.x + state.viewport_pan.x;
        out_image_pos.y = canvas_pos.y + state.viewport_pan.y;
    }
}

// Handle mouse interaction for viewport
void HandleViewportInteraction(const ImVec2& canvas_pos, const ImVec2& canvas_size, ApplicationState& state) {
    ImGuiIO& io = ImGui::GetIO();
    
    // Check if mouse is over canvas
    ImVec2 mouse_pos = io.MousePos;
    bool mouse_over_canvas = (mouse_pos.x >= canvas_pos.x && mouse_pos.x <= canvas_pos.x + canvas_size.x &&
                             mouse_pos.y >= canvas_pos.y && mouse_pos.y <= canvas_pos.y + canvas_size.y);
    
    if (mouse_over_canvas) {
        // Mouse wheel for zooming
        if (io.MouseWheel != 0.0f && !state.fit_to_window) {
            float zoom_factor = 1.0f + io.MouseWheel * 0.1f;
            state.viewport_zoom = std::clamp(state.viewport_zoom * zoom_factor, 0.1f, 5.0f);
        }
        
        // Middle mouse button for panning
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) && !state.fit_to_window) {
            ImVec2 delta = io.MouseDelta;
            state.viewport_pan.x += delta.x;
            state.viewport_pan.y += delta.y;
        }
        
        // Double-click to fit to window
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            state.fit_to_window = !state.fit_to_window;
            if (state.fit_to_window) {
                state.viewport_pan = ImVec2(0, 0);
            }
        }
        
        // Right-click context menu
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            ImGui::OpenPopup("ViewportContextMenu");
        }
    }
    
    // Context menu
    if (ImGui::BeginPopup("ViewportContextMenu")) {
        if (ImGui::MenuItem("Fit to Window")) {
            state.fit_to_window = true;
            state.viewport_pan = ImVec2(0, 0);
        }
        if (ImGui::MenuItem("Actual Size (100%)")) {
            state.fit_to_window = false;
            state.viewport_zoom = 1.0f;
            state.viewport_pan = ImVec2(0, 0);
        }
        if (ImGui::MenuItem("Reset Pan")) {
            state.viewport_pan = ImVec2(0, 0);
        }
        ImGui::EndPopup();
    }
}

// Draw overlays on the viewport
void DrawViewportOverlays(ImDrawList* draw_list, const ImVec2& image_pos, const ImVec2& image_size, ApplicationState& state) {
    if (state.show_grid) {
        // Draw grid overlay
        ImU32 grid_color = IM_COL32(255, 255, 255, 100);
        float grid_spacing = 50.0f * state.dpi_scale;
        
        for (float x = image_pos.x; x < image_pos.x + image_size.x; x += grid_spacing) {
            draw_list->AddLine(ImVec2(x, image_pos.y), ImVec2(x, image_pos.y + image_size.y), grid_color);
        }
        for (float y = image_pos.y; y < image_pos.y + image_size.y; y += grid_spacing) {
            draw_list->AddLine(ImVec2(image_pos.x, y), ImVec2(image_pos.x + image_size.x, y), grid_color);
        }
    }
    
    if (state.show_camera_frustums) {
        // Draw camera frustum indicators
        draw_list->AddText(ImVec2(image_pos.x + 10, image_pos.y + 10),
                          IM_COL32(255, 255, 0, 255), "Camera Frustums");
    }
}

// Draw placeholder when stitching not ready
void DrawViewportPlaceholder(ImDrawList* draw_list, const ImVec2& canvas_pos, const ImVec2& canvas_size, ApplicationState& state) {
    const char* status_text;
    if (!state.stitching_initialized) {
        status_text = "INITIALIZE STITCHING FIRST";
    } else if (state.stitched_output_texture == 0) {
        status_text = "NO OUTPUT TEXTURE - TRY MANUAL UPDATE";
    } else {
        status_text = "3D REPROJECTION READY";
    }
    
    ImVec2 text_size = ImGui::CalcTextSize(status_text);
    ImVec2 text_pos = ImVec2(
        canvas_pos.x + (canvas_size.x - text_size.x) * 0.5f,
        canvas_pos.y + (canvas_size.y - text_size.y) * 0.5f
    );
    
    draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), status_text);
}

// Bottom controls for the viewport
void DrawViewportBottomControls(ApplicationState& state) {
    ImGui::Separator();
    
    // Zoom controls
    ImGui::Text("Zoom:");
    ImGui::SameLine();
    if (ImGui::Button("-")) {
        state.viewport_zoom = std::max(0.1f, state.viewport_zoom * 0.8f);
        state.fit_to_window = false;
    }
    ImGui::SameLine();
    ImGui::Text("%.0f%%", state.viewport_zoom * 100);
    ImGui::SameLine();
    if (ImGui::Button("+")) {
        state.viewport_zoom = std::min(5.0f, state.viewport_zoom * 1.25f);
        state.fit_to_window = false;
    }
    
    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();
    
    // Resolution control
    const char* resolutions[] = {"720p", "1080p", "4K"};
    int current_res = 1; // Default to 1080p
    if (strcmp(state.current_output_resolution, "720p") == 0) current_res = 0;
    else if (strcmp(state.current_output_resolution, "4K") == 0) current_res = 2;
    
    ImGui::Text("Res:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##resolution", &current_res, resolutions, IM_ARRAYSIZE(resolutions))) {
        strcpy(state.current_output_resolution, resolutions[current_res]);
        if (state.stitching_initialized) {
            state.UpdateStitching();
        }
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Refresh")) {
        if (state.stitching_initialized) {
            state.UpdateStitching();
        }
    }
}

// Updated main viewport to show stitched output
void DrawStitchedMainViewport(ApplicationState& state) {
    // Option 1: Dockable viewport window
    ImGuiWindowFlags viewport_flags = ImGuiWindowFlags_NoScrollbar;
    
    // Make window dockable and resizable
    if (state.viewport_dockable) {
        viewport_flags |= ImGuiWindowFlags_NoCollapse;
    }
    
    // Optional: Make it a child window that can be embedded anywhere
    if (state.viewport_embedded) {
        ImGui::BeginChild("EmbeddedViewport", ImVec2(0, 0), true, viewport_flags);
    } else {
        // Regular movable window
        ImGui::Begin("3D Reprojection Output", &state.show_stitched_output, viewport_flags);
    }
    
    // Add viewport controls at the top
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Fit to Window", nullptr, &state.fit_to_window);
            ImGui::MenuItem("Maintain Aspect", nullptr, &state.maintain_aspect);
            ImGui::Separator();
            
            const char* zoom_levels[] = {"25%", "50%", "75%", "100%", "125%", "150%", "200%"};
            float zoom_values[] = {0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f};
            
            for (int i = 0; i < IM_ARRAYSIZE(zoom_levels); i++) {
                if (ImGui::MenuItem(zoom_levels[i], nullptr, abs(state.viewport_zoom - zoom_values[i]) < 0.01f)) {
                    state.viewport_zoom = zoom_values[i];
                    state.fit_to_window = false;
                }
            }
            
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Display")) {
            ImGui::MenuItem("Show Grid", nullptr, &state.show_grid);
            ImGui::MenuItem("Show Overlaps", nullptr, &state.show_overlaps);
            ImGui::MenuItem("Show Camera Frustums", nullptr, &state.show_camera_frustums);
            ImGui::EndMenu();
        }
        
        ImGui::EndMenuBar();
    }
    
    // Calculate display area
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    
    // Reserve space for bottom controls if needed
    if (state.show_viewport_controls) {
        canvas_size.y -= 60 * state.dpi_scale;
    }
    
    if (canvas_size.x > 0 && canvas_size.y > 0) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        // Add background
        draw_list->AddRectFilled(canvas_pos, 
            ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
            IM_COL32(30, 30, 40, 255));
        
        if (state.stitching_initialized && state.stitched_output_texture != 0) {
            // Calculate image display size and position
            ImVec2 image_size, image_pos;
            CalculateImageDisplayTransform(canvas_size, canvas_pos, state, image_size, image_pos);
            
            // Display the stitched output
            draw_list->AddImage(
                (ImTextureID)(intptr_t)state.stitched_output_texture,
                image_pos,
                ImVec2(image_pos.x + image_size.x, image_pos.y + image_size.y),
                ImVec2(0, 1), ImVec2(1, 0) // Flip Y for OpenGL
            );
            
            // Add optional overlays
            DrawViewportOverlays(draw_list, image_pos, image_size, state);
            
        } else {
            // Show status when not ready
            DrawViewportPlaceholder(draw_list, canvas_pos, canvas_size, state);
        }
        
        // Handle mouse interaction for panning/zooming
        HandleViewportInteraction(canvas_pos, canvas_size, state);
    }
    
    // Bottom controls
    if (state.show_viewport_controls) {
        DrawViewportBottomControls(state);
    }
    
    if (state.viewport_embedded) {
        ImGui::EndChild();
    } else {
        ImGui::End();
    }
}

// Stitching debug window
void DrawStitchingDebugWindow(ApplicationState& state) {
    if (!state.show_stitching_debug) return;
    
    ImGui::Begin("Stitching Debug", &state.show_stitching_debug);
    
    if (!state.stitching_initialized) {
        ImGui::Text("Stitching not initialized");
        ImGui::End();
        return;
    }
    
    // Camera information
    if (ImGui::CollapsingHeader("Camera Information", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto& cameras = state.stitching_pipeline->GetCameras();
        
        for (size_t i = 0; i < cameras.size(); i++) {
            ImGui::PushID(i);
            
            if (ImGui::TreeNode(cameras[i].name.c_str())) {
                const auto& intrinsics = cameras[i].intrinsics;
                const auto& extrinsics = cameras[i].extrinsics;
                
                ImGui::Text("Image Size: %dx%d", intrinsics.image_width, intrinsics.image_height);
                
                ImGui::Text("Camera Matrix:");
                for (int row = 0; row < 3; row++) {
                    ImGui::Text("  [%.2f, %.2f, %.2f]", 
                               intrinsics.camera_matrix[0][row],
                               intrinsics.camera_matrix[1][row], 
                               intrinsics.camera_matrix[2][row]);
                }
                
                ImGui::Text("Distortion: k1=%.4f, k2=%.4f, p1=%.4f, p2=%.4f",
                           intrinsics.distortion_coeffs[0],
                           intrinsics.distortion_coeffs[1],
                           intrinsics.distortion_coeffs[2],
                           intrinsics.distortion_coeffs[3]);
                
                ImGui::Text("Translation: [%.3f, %.3f, %.3f]",
                           extrinsics.tvec.x, extrinsics.tvec.y, extrinsics.tvec.z);
                
                ImGui::TreePop();
            }
            
            ImGui::PopID();
        }
    }
    
    // World coordinate system info
    if (ImGui::CollapsingHeader("World Coordinate System")) {
        const auto& world_config = state.stitching_pipeline->GetWorldConfig();
        
        ImGui::Text("Bounds: [%.1f, %.1f] to [%.1f, %.1f]",
                   world_config.bounds_min.x, world_config.bounds_min.y,
                   world_config.bounds_max.x, world_config.bounds_max.y);
        ImGui::Text("Z-Plane: %.2f", world_config.z_plane);
        ImGui::Text("Output: %dx%d", world_config.output_width, world_config.output_height);
        ImGui::Text("Pixel Ratio: %.1f px/m", world_config.pixel_to_meter_ratio);
    }
    
    // Coverage map visualization
    if (ImGui::CollapsingHeader("Coverage Map") && state.stitching_pipeline) {
        static std::vector<float> coverage_map;
        static bool map_computed = false;
        
        if (ImGui::Button("Compute Coverage Map") || !map_computed) {
            coverage_map = state.stitching_pipeline->GetCoverageMap();
            map_computed = true;
        }
        
        if (!coverage_map.empty()) {
            ImGui::Text("Coverage map: 200x120 pixels");
            ImGui::Text("Green = full coverage, Red = no coverage");
            
            // TODO: Visualize coverage map as a heatmap
            float min_coverage = *std::min_element(coverage_map.begin(), coverage_map.end());
            float max_coverage = *std::max_element(coverage_map.begin(), coverage_map.end());
            ImGui::Text("Coverage range: %.2f to %.2f", min_coverage, max_coverage);
        }
    }
    
    ImGui::End();
}

// Update the main menu bar to include stitching options
void DrawMainMenuBarWithStitching(ApplicationState& state) {
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Load Calibration")) {
            state.show_calibration_loader = true;
        }
        if (ImGui::MenuItem("Load Test Images")) {
            // TODO: Implement test image loader dialog
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Save Stitching Settings")) {
            // TODO: Implement settings save
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Exit")) {
            // TODO: Implement proper exit
        }
        ImGui::EndMenu();
    }
    
    if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem("Show Debug Info", nullptr, &state.show_debug_window);
        ImGui::MenuItem("Show Performance", nullptr, &state.show_performance_window);
        ImGui::MenuItem("Show Stitching Debug", nullptr, &state.show_stitching_debug);
        ImGui::Separator();
        if (ImGui::MenuItem("Fullscreen")) {
            // TODO: Implement fullscreen toggle
        }
        ImGui::EndMenu();
    }
    
    if (ImGui::BeginMenu("Stitching")) {
        if (ImGui::MenuItem("Initialize Pipeline")) {
            state.InitializeStitching();
        }
        if (ImGui::MenuItem("Reset Pipeline")) {
            state.ResetStitching();
        }
        ImGui::Separator();
        ImGui::MenuItem("Show Camera Frustums", nullptr, &state.show_camera_frustums);
        ImGui::MenuItem("Show Overlap Regions", nullptr, &state.show_overlap_regions);
        ImGui::MenuItem("Show Coverage Map", nullptr, &state.show_coverage_map);
        ImGui::Separator();
        if (ImGui::MenuItem("Recalibrate")) {
            // TODO: Implement recalibration
        }
        ImGui::EndMenu();
    }
    
    if (ImGui::BeginMenu("Tools")) {
        if (ImGui::MenuItem("Camera Setup")) {
            state.show_camera_setup = true;
        }
        if (ImGui::MenuItem("Settings")) {
            state.show_settings = true;
        }
        ImGui::EndMenu();
    }
    
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("About")) {
            state.show_about = true;
        }
        ImGui::EndMenu();
    }
}

// Performance monitoring window
void DrawPerformanceWindow(ApplicationState& state) {
    if (!state.show_performance_window) return;
    
    ImGui::Begin("Performance", &state.show_performance_window);
    
    // Simulate performance data updates
    static float time_acc = 0.0f;
    time_acc += ImGui::GetIO().DeltaTime;
    if (time_acc > 0.1f) { // Update every 100ms
        state.gpu_time = 40.0f + sin(GetTime()) * 5.0f;
        state.cpu_time = 10.0f + cos(GetTime() * 1.5f) * 3.0f;
        state.fps = 30.0f - sin(GetTime() * 0.8f) * 2.0f;
        state.memory_usage = 2.0f + sin(GetTime() * 0.3f) * 0.2f;
        time_acc = 0.0f;
    }
    
    ImGui::Text("GPU: %.1fms", state.gpu_time);
    ImGui::Text("CPU: %.1fms", state.cpu_time);
    ImGui::Text("FPS: %.1f", state.fps);
    ImGui::Text("Memory: %.1fGB", state.memory_usage);
    
    // Performance graph
    static float fps_history[100] = {};
    static int fps_history_offset = 0;
    fps_history[fps_history_offset] = state.fps;
    fps_history_offset = (fps_history_offset + 1) % IM_ARRAYSIZE(fps_history);
    
    ImGui::PlotLines("FPS", fps_history, IM_ARRAYSIZE(fps_history), fps_history_offset, 
                     nullptr, 0.0f, 35.0f, ImVec2(0, 80 * state.dpi_scale));
    
    ImGui::End();
}

// Debug information window
void DrawDebugWindow(ApplicationState& state) {
    if (!state.show_debug_window) return;
    
    ImGui::Begin("Debug Info", &state.show_debug_window);
    
    ImGui::Text("Calibration: %s", state.calibration_ok ? "OK" : "FAIL");
    ImGui::Text("Overlap A-B: %s", state.overlap_ab_ok ? "OK" : "FAIL");
    ImGui::Text("Overlap B-C: %s", state.overlap_bc_ok ? "OK" : "FAIL");
    ImGui::Text("Sync Offset: %dms", state.sync_offset);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("System Info:");
    ImGui::Text("DPI Scale: %.2fx", state.dpi_scale);
    ImGui::Text("Display: %.0fx%.0f", ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y);
    
    ImGui::End();
}

// Modal dialogs
void DrawModalDialogs(ApplicationState& state) {
    // Camera Setup Modal
    if (state.show_camera_setup) {
        ImGui::OpenPopup("Camera Setup");
        state.show_camera_setup = false;
    }
    
    if (ImGui::BeginPopupModal("Camera Setup", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Individual camera configuration would go here");
        ImGui::Spacing();
        
        if (ImGui::Button("OK", ImVec2(120 * state.dpi_scale, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120 * state.dpi_scale, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    
    // About Dialog
    if (state.show_about) {
        ImGui::OpenPopup("About");
        state.show_about = false;
    }
    
    if (ImGui::BeginPopupModal("About", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Stitching Application v1.0");
        ImGui::Text("Professional real-time camera stitching");
        ImGui::Spacing();
        ImGui::Text("Built with Dear ImGui and OpenGL");
        ImGui::Spacing();
        
        if (ImGui::Button("Close", ImVec2(120 * state.dpi_scale, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

// Camera control panel
void DrawCameraControls(ApplicationState& state) {
    ImGui::Begin("Camera Controls");
    
    // Camera Status
    if (ImGui::CollapsingHeader("Camera Status", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (int i = 0; i < 3; i++) {
            ImGui::PushID(i);
            
            // Status indicator
            ImVec4 status_color = state.camera_connected[i] ? 
                ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
            
            ImGui::TextColored(status_color, "%s %s: %.1f fps", 
                state.camera_connected[i] ? "[OK]" : "[ERR]",
                state.camera_names[i], 
                state.camera_fps[i]);
            
            ImGui::PopID();
        }
        
        ImGui::Spacing();
        if (ImGui::Button("Reconnect All", ImVec2(-1, 0))) {
            // TODO: Implement reconnect
        }
        if (ImGui::Button("Individual Setup", ImVec2(-1, 0))) {
            state.show_camera_setup = true;
        }
    }
    
    // Camera Settings
    if (ImGui::CollapsingHeader("Camera Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Exposure", &state.exposure, 0.0f, 1.0f);
        ImGui::SliderFloat("Gain", &state.gain, 0.0f, 1.0f);
        ImGui::Checkbox("Auto WB", &state.auto_wb);
        
        const char* sync_modes[] = {"Hardware", "Software", "Manual"};
        ImGui::Combo("Sync Mode", &state.sync_mode, sync_modes, IM_ARRAYSIZE(sync_modes));
    }
    
    ImGui::End();
}