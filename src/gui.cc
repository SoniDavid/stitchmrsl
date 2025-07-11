#include "gui.hh"
#include <iostream>
#include <fstream>

// --- HELPER FUNCTION ---
void MatToTexture(const cv::Mat& mat, GLuint& texture_id) {
    if (mat.empty()) return;

    if (texture_id == 0) {
        glGenTextures(1, &texture_id);
    }

    glBindTexture(GL_TEXTURE_2D, texture_id);

    // Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Convert BGR to RGB and ensure continuous memory layout
    cv::Mat rgb_mat;
    cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    
    // Ensure the image is continuous in memory
    if (!rgb_mat.isContinuous()) {
        rgb_mat = rgb_mat.clone();
    }

    // Upload pixels with correct alignment
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_mat.cols, rgb_mat.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_mat.data);
}

// --- APPLICATION STATE METHODS ---
void ApplicationState::InitializeStitching() {
    pipeline = std::make_unique<StitchingPipeline>();
    
    calibration_loaded = pipeline->LoadIntrinsicsData(intrinsics_file_path) &&
                         pipeline->LoadExtrinsicsData(extrinsics_file_path);

    if (use_rtsp_streams) {
        test_images_loaded = CaptureRTSPFrames();
    } else {
        std::vector<std::string> paths;
        for(int i=0; i<3; ++i) paths.push_back(test_image_paths[i]);
        test_images_loaded = pipeline->LoadTestImages(paths);
    }

    if (calibration_loaded && test_images_loaded) {
        stitching_initialized = true;
        status_message = "Ready. Click 'Create Panorama'.";
        
        // Apply manual adjustments if they were loaded from settings
        ApplyInitialAdjustments();
    } else {
        stitching_initialized = false;
        status_message = "Failed to load files. Check paths.";
    }
}

void ApplicationState::CreatePanorama() {
    if (!stitching_initialized) {
        status_message = "Cannot create panorama, not initialized.";
        return;
    }
    status_message = "Processing...";
    
    // Use manual adjustments if enabled, otherwise use precomputed
    if (manual_adjustments_enabled) {
        UpdatePanoramaWithAdjustments();
    } else {
        pipeline->SetBlendingMode(static_cast<BlendingMode>(selected_blending_mode));
        stitched_mat = pipeline->CreatePanoramaFromPrecomputed();

        if (!stitched_mat.empty()) {
            MatToTexture(stitched_mat, stitched_texture_id);
            status_message = "Panorama created successfully!";
            // Save the result automatically
            cv::imwrite("panorama_result.jpg", stitched_mat);
        } else {
            status_message = "Panorama creation failed.";
        }
    }
}

void ApplicationState::ResetStitching() {
    pipeline.reset();
    stitching_initialized = false;
    calibration_loaded = false;
    test_images_loaded = false;
    stitched_mat.release();
    if (stitched_texture_id != 0) {
        glDeleteTextures(1, &stitched_texture_id);
        stitched_texture_id = 0;
    }
    status_message = "Pipeline reset. Initialize to begin.";
}

void ApplicationState::UpdatePanoramaWithAdjustments() {
    if (!stitching_initialized) return;
    
    pipeline->SetBlendingMode(static_cast<BlendingMode>(selected_blending_mode));
    
    if (manual_adjustments_enabled) {
        // Create custom transform matrices from adjustment parameters
        
        // AB Transform (izquierda -> central)
        cv::Mat ab_custom = cv::Mat::eye(3, 3, CV_64F);
        float ab_rad = ab_rotation_deg * CV_PI / 180.0f; //ab_rotation_deg=0.80
        float ab_cos = cos(ab_rad);
        float ab_sin = sin(ab_rad);
        
        ab_custom.at<double>(0, 0) = ab_cos * ab_scale_x; // ab_scale_x = 0.9877
        ab_custom.at<double>(0, 1) = -ab_sin * ab_scale_x;
        ab_custom.at<double>(0, 2) = ab_translation_x; //-2.8
        ab_custom.at<double>(1, 0) = ab_sin * ab_scale_y;  // ab_scale_y = 1.000
        ab_custom.at<double>(1, 1) = ab_cos * ab_scale_y;
        ab_custom.at<double>(1, 2) = ab_translation_y; //0.0
        
        // BC Transform (derecha -> central)
        cv::Mat bc_custom = cv::Mat::eye(3, 3, CV_64F);
        float bc_rad = bc_rotation_deg * CV_PI / 180.0f;
        float bc_cos = cos(bc_rad);
        float bc_sin = sin(bc_rad);
        
        bc_custom.at<double>(0, 0) = bc_cos * bc_scale_x;
        bc_custom.at<double>(0, 1) = -bc_sin * bc_scale_x;
        bc_custom.at<double>(0, 2) = bc_translation_x; //21.1
        bc_custom.at<double>(1, 0) = bc_sin * bc_scale_y;
        bc_custom.at<double>(1, 1) = bc_cos * bc_scale_y;
        bc_custom.at<double>(1, 2) = bc_translation_y; // 0.00
        
        stitched_mat = pipeline->CreatePanoramaWithCustomTransforms(ab_custom, bc_custom);
    } else {
        stitched_mat = pipeline->CreatePanoramaFromPrecomputed();
    }
    
    if (!stitched_mat.empty()) {
        MatToTexture(stitched_mat, stitched_texture_id);
        status_message = "Panorama updated with adjustments!";
    } else {
        status_message = "Failed to update panorama.";
    }
}

void ApplicationState::ApplyInitialAdjustments() {
    // Apply manual adjustments that were loaded from ini file if enabled
    if (manual_adjustments_enabled && stitching_initialized) {
        UpdatePanoramaWithAdjustments();
    }
}

// Global pointer to access ApplicationState from settings handlers
static ApplicationState* g_app_state = nullptr;

void* ManualAdjustments_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name) {
    if (strcmp(name, "Data") == 0) {
        return g_app_state; // Return our app state as the entry
    }
    return nullptr;
}

bool ApplicationState::CaptureRTSPFrames() {
    bool all_success = true;
    std::vector<cv::Mat> frames;

    for (int i = 0; i < 3; ++i) {
        if (!rtsp_caps[i].isOpened()) {
            rtsp_caps[i].open(rtsp_urls[i]);
            if (!rtsp_caps[i].isOpened()) {
                status_message = "Failed to open RTSP stream: " + std::string(rtsp_urls[i]);
                all_success = false;
                break;
            }
        }

        cv::Mat frame;
        if (!rtsp_caps[i].read(frame) || frame.empty()) {
            status_message = "Failed to read frame from RTSP: " + std::string(rtsp_urls[i]);
            all_success = false;
            break;
        }
        frames.push_back(frame);
    }

    if (all_success) {
        test_images_loaded = pipeline->LoadTestImagesFromMats(frames);  // necesitas implementar este método
        status_message = "RTSP frames captured and loaded!";
    }

    return all_success;
}

void ApplicationState::UpdateRTSPFramesAndPanorama() {
    if (!use_rtsp_streams || !stitching_initialized) return;

    std::vector<cv::Mat> frames;
    bool all_success = true;

    for (int i = 0; i < 3; ++i) {
        cv::Mat frame;
        if (!rtsp_caps[i].read(frame) || frame.empty()) {
            status_message = "Failed to read frame from RTSP stream: " + std::string(rtsp_urls[i]);
            all_success = false;
            break;
        }
        frames.push_back(frame);
    }

    if (all_success) {
        pipeline->SetBlendingMode(static_cast<BlendingMode>(selected_blending_mode));
        pipeline->LoadTestImagesFromMats(frames);
        stitched_mat = pipeline->CreatePanoramaFromPrecomputed();
        if (!stitched_mat.empty()) {
            MatToTexture(stitched_mat, stitched_texture_id);
        }
    }
}


void ManualAdjustments_ReadLine(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line) {
    if (!entry || !g_app_state) return;
    
    // Parse key=value pairs
    const char* eq_pos = strchr(line, '=');
    if (!eq_pos) return;
    
    size_t key_len = eq_pos - line;
    const char* value_str = eq_pos + 1;
    
    if (strncmp(line, "manual_enabled", key_len) == 0) {
        g_app_state->manual_adjustments_enabled = (strcmp(value_str, "1") == 0);
    } else if (strncmp(line, "auto_update", key_len) == 0) {
        g_app_state->auto_update_enabled = (strcmp(value_str, "1") == 0);
    } else if (strncmp(line, "ab_tx", key_len) == 0) {
        g_app_state->ab_translation_x = (float)atof(value_str);
    } else if (strncmp(line, "ab_ty", key_len) == 0) {
        g_app_state->ab_translation_y = (float)atof(value_str);
    } else if (strncmp(line, "ab_rot", key_len) == 0) {
        g_app_state->ab_rotation_deg = (float)atof(value_str);
    } else if (strncmp(line, "ab_sx", key_len) == 0) {
        g_app_state->ab_scale_x = (float)atof(value_str);
    } else if (strncmp(line, "ab_sy", key_len) == 0) {
        g_app_state->ab_scale_y = (float)atof(value_str);
    } else if (strncmp(line, "bc_tx", key_len) == 0) {
        g_app_state->bc_translation_x = (float)atof(value_str);
    } else if (strncmp(line, "bc_ty", key_len) == 0) {
        g_app_state->bc_translation_y = (float)atof(value_str);
    } else if (strncmp(line, "bc_rot", key_len) == 0) {
        g_app_state->bc_rotation_deg = (float)atof(value_str);
    } else if (strncmp(line, "bc_sx", key_len) == 0) {
        g_app_state->bc_scale_x = (float)atof(value_str);
    } else if (strncmp(line, "bc_sy", key_len) == 0) {
        g_app_state->bc_scale_y = (float)atof(value_str);
    }
}

void ManualAdjustments_WriteAll(ImGuiContext*, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
    if (!g_app_state) return;
    
    buf->appendf("[%s][Data]\n", handler->TypeName);
    buf->appendf("manual_enabled=%d\n", g_app_state->manual_adjustments_enabled ? 1 : 0);
    buf->appendf("auto_update=%d\n", g_app_state->auto_update_enabled ? 1 : 0);
    buf->appendf("ab_tx=%.3f\n", g_app_state->ab_translation_x);
    buf->appendf("ab_ty=%.3f\n", g_app_state->ab_translation_y);
    buf->appendf("ab_rot=%.3f\n", g_app_state->ab_rotation_deg);
    buf->appendf("ab_sx=%.6f\n", g_app_state->ab_scale_x);
    buf->appendf("ab_sy=%.6f\n", g_app_state->ab_scale_y);
    buf->appendf("bc_tx=%.3f\n", g_app_state->bc_translation_x);
    buf->appendf("bc_ty=%.3f\n", g_app_state->bc_translation_y);
    buf->appendf("bc_rot=%.3f\n", g_app_state->bc_rotation_deg);
    buf->appendf("bc_sx=%.6f\n", g_app_state->bc_scale_x);
    buf->appendf("bc_sy=%.6f\n", g_app_state->bc_scale_y);
    buf->append("\n");
}

void RegisterManualAdjustmentsHandler(ApplicationState* app_state) {
    g_app_state = app_state;
    
    ImGuiSettingsHandler ini_handler;
    ini_handler.TypeName = "ManualAdjustments";
    ini_handler.TypeHash = ImHashStr("ManualAdjustments");
    ini_handler.ReadOpenFn = ManualAdjustments_ReadOpen;
    ini_handler.ReadLineFn = ManualAdjustments_ReadLine;
    ini_handler.WriteAllFn = ManualAdjustments_WriteAll;
    ImGui::AddSettingsHandler(&ini_handler);
}

// --- GUI DRAWING FUNCTIONS ---

void DrawStitchingSetupPanel(ApplicationState& state) {
    ImGui::Begin("Setup");

    ImGui::Text("1. Calibration Files");
    ImGui::InputText("Intrinsics JSON", state.intrinsics_file_path, sizeof(state.intrinsics_file_path));
    ImGui::InputText("Extrinsics JSON", state.extrinsics_file_path, sizeof(state.extrinsics_file_path));

    ImGui::Separator();
    ImGui::Text("2. Test Images");
    for (int i = 0; i < 3; i++) {
        ImGui::PushID(i);
        ImGui::InputText(state.camera_names[i], state.test_image_paths[i], sizeof(state.test_image_paths[i]));
        ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::Text("Blending Options");
    ImGui::Combo("Mode", &state.selected_blending_mode, state.blending_modes, IM_ARRAYSIZE(state.blending_modes));

    ImGui::Separator();
    ImGui::Text("Viewer Orientation");
    
    // Four orientation buttons
    if (ImGui::Button("Normal (0°)")) {
        state.rotation_degrees = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Right (90°)")) {
        state.rotation_degrees = 90.0f;
    }
    
    if (ImGui::Button("Inverted (180°)")) {
        state.rotation_degrees = 180.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Left (-90°)")) {
        state.rotation_degrees = -90.0f;
    }
    
    // Show current rotation
    ImGui::Text("Current: %.1f°", state.rotation_degrees);

    ImGui::Separator();
    ImGui::Text("3. Actions");
    if (!state.stitching_initialized) {
        if (ImGui::Button("Initialize Pipeline", ImVec2(-1, 0))) {
            state.InitializeStitching();
        }
    } else {
        if (ImGui::Button("Create Panorama", ImVec2(-1, 0))) {
            state.CreatePanorama();
        }
        if (ImGui::Button("Reset", ImVec2(-1, 0))) {
            state.ResetStitching();
        }
    }
    
    ImGui::End();
}

void DrawImageViewerPanel(ApplicationState& state) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Panorama Viewer");

    if (state.stitched_texture_id != 0 && !state.stitched_mat.empty()) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_p0 = ImGui::GetCursorScreenPos(); // Top-left of the panel
        ImVec2 canvas_sz = ImGui::GetContentRegionAvail(); // Size of the panel

        // Aspect ratio calculation
        float aspect_ratio = (float)state.stitched_mat.cols / (float)state.stitched_mat.rows;
        ImVec2 display_size = canvas_sz;
        if (canvas_sz.x / aspect_ratio > canvas_sz.y) {
            display_size.x = canvas_sz.y * aspect_ratio;
        } else {
            display_size.y = canvas_sz.x / aspect_ratio;
        }

        // Center the image within the panel
        ImVec2 center_pos = ImVec2(canvas_p0.x + canvas_sz.x * 0.5f, canvas_p0.y + canvas_sz.y * 0.5f);
        ImVec2 top_left = ImVec2(center_pos.x - display_size.x * 0.5f, center_pos.y - display_size.y * 0.5f);

        // Rotation logic
        float rad = state.rotation_degrees * 3.14159265f / 180.0f;
        float cos_a = cos(rad);
        float sin_a = sin(rad);
        ImVec2 pos[4] = {
            top_left,
            ImVec2(top_left.x + display_size.x, top_left.y),
            ImVec2(top_left.x + display_size.x, top_left.y + display_size.y),
            ImVec2(top_left.x, top_left.y + display_size.y)
        };

        for (int i = 0; i < 4; i++) {
            ImVec2 p = pos[i];
            pos[i].x = center_pos.x + (p.x - center_pos.x) * cos_a - (p.y - center_pos.y) * sin_a;
            pos[i].y = center_pos.y + (p.x - center_pos.x) * sin_a + (p.y - center_pos.y) * cos_a;
        }

        // UV coordinates for the texture (always in order: top-left, top-right, bottom-right, bottom-left)
        ImVec2 uv[4] = {
            ImVec2(0.0f, 0.0f), // top-left
            ImVec2(1.0f, 0.0f), // top-right
            ImVec2(1.0f, 1.0f), // bottom-right
            ImVec2(0.0f, 1.0f)  // bottom-left
        };
        
        draw_list->AddImageQuad((void*)(intptr_t)state.stitched_texture_id, 
                                pos[0], pos[1], pos[2], pos[3],
                                uv[0], uv[1], uv[2], uv[3]);

    } else {
        ImGui::TextWrapped("Output will be displayed here. The final image will also be saved as 'panorama_result.jpg' in the executable's directory.");
    }

    ImGui::End();
    ImGui::PopStyleVar();
}

void DrawStatusPanel(ApplicationState& state) {
    ImGui::Begin("Status");

    ImVec4 green(0.0f, 1.0f, 0.0f, 1.0f);
    ImVec4 red(1.0f, 0.0f, 0.0f, 1.0f);

    ImGui::Text("Calibration:"); ImGui::SameLine();
    ImGui::TextColored(state.calibration_loaded ? green : red, state.calibration_loaded ? "Loaded" : "Not Loaded");

    ImGui::Text("Test Images:"); ImGui::SameLine();
    ImGui::TextColored(state.test_images_loaded ? green : red, state.test_images_loaded ? "Loaded" : "Not Loaded");
    
    ImGui::Separator();
    ImGui::Text("Current Status:");
    ImGui::TextWrapped("%s", state.status_message.c_str());

    ImGui::End();
}

void DrawManualAdjustmentsPanel(ApplicationState& state) {
    ImGui::Begin("Manual Adjustments");
    
    // Main toggle for manual adjustments
    if (ImGui::Checkbox("Enable Manual Adjustments", &state.manual_adjustments_enabled)) {
        if (state.auto_update_enabled && state.stitching_initialized) {
            state.UpdatePanoramaWithAdjustments();
        }
    }
    
    ImGui::SameLine();
    ImGui::Checkbox("Auto Update", &state.auto_update_enabled);
    
    if (!state.manual_adjustments_enabled) {
        ImGui::BeginDisabled();
    }
    
    ImGui::Separator();
    ImGui::Text("Blending Mode (Real-time)");
    if (ImGui::Combo("##BlendingMode", &state.selected_blending_mode, state.blending_modes, IM_ARRAYSIZE(state.blending_modes))) {
        if (state.auto_update_enabled && state.stitching_initialized) {
            state.UpdatePanoramaWithAdjustments();
        }
    }
    
    ImGui::Separator();
    ImGui::Text("AB Transform (Izquierda -> Central)");
    
    bool ab_changed = false;
    
    if (ImGui::DragFloat("AB Translation X", &state.ab_translation_x, 0.1f, -200.0f, 200.0f, "%.1f px")) {
        ab_changed = true;
    }
    if (ImGui::DragFloat("AB Translation Y", &state.ab_translation_y, 0.1f, -200.0f, 200.0f, "%.1f px")) {
        ab_changed = true;
    }
    if (ImGui::DragFloat("AB Rotation", &state.ab_rotation_deg, 0.01f, -10.0f, 10.0f, "%.2f deg")) {
        ab_changed = true;
    }
    if (ImGui::DragFloat("AB Scale X", &state.ab_scale_x, 0.0001f, 0.9f, 1.1f, "%.4f")) {
        ab_changed = true;
    }
    if (ImGui::DragFloat("AB Scale Y", &state.ab_scale_y, 0.0001f, 0.9f, 1.1f, "%.4f")) {
        ab_changed = true;
    }
    
    if (ImGui::Button("Reset AB")) {
        state.ab_translation_x = 0.0f;
        state.ab_translation_y = 0.0f;
        state.ab_rotation_deg = 0.0f;
        state.ab_scale_x = 1.0f;
        state.ab_scale_y = 1.0f;
        ab_changed = true;
    }
    
    ImGui::Separator();
    ImGui::Text("BC Transform (Derecha -> Central)");
    
    bool bc_changed = false;
    
    if (ImGui::DragFloat("BC Translation X", &state.bc_translation_x, 0.1f, -200.0f, 200.0f, "%.1f px")) {
        bc_changed = true;
    }
    if (ImGui::DragFloat("BC Translation Y", &state.bc_translation_y, 0.1f, -200.0f, 200.0f, "%.1f px")) {
        bc_changed = true;
    }
    if (ImGui::DragFloat("BC Rotation", &state.bc_rotation_deg, 0.01f, -10.0f, 10.0f, "%.2f deg")) {
        bc_changed = true;
    }
    if (ImGui::DragFloat("BC Scale X", &state.bc_scale_x, 0.0001f, 0.9f, 1.1f, "%.4f")) {
        bc_changed = true;
    }
    if (ImGui::DragFloat("BC Scale Y", &state.bc_scale_y, 0.0001f, 0.9f, 1.1f, "%.4f")) {
        bc_changed = true;
    }
    
    if (ImGui::Button("Reset BC")) {
        state.bc_translation_x = 0.0f;
        state.bc_translation_y = 0.0f;
        state.bc_rotation_deg = 0.0f;
        state.bc_scale_x = 1.0f;
        state.bc_scale_y = 1.0f;
        bc_changed = true;
    }
    
    ImGui::Separator();
    if (ImGui::Button("Reset All Adjustments", ImVec2(-1, 0))) {
        state.ab_translation_x = 0.0f;
        state.ab_translation_y = 0.0f;
        state.ab_rotation_deg = 0.0f;
        state.ab_scale_x = 1.0f;
        state.ab_scale_y = 1.0f;
        state.bc_translation_x = 0.0f;
        state.bc_translation_y = 0.0f;
        state.bc_rotation_deg = 0.0f;
        state.bc_scale_x = 1.0f;
        state.bc_scale_y = 1.0f;
        ab_changed = bc_changed = true;
    }
    
    if (ImGui::Button("Manual Update", ImVec2(-1, 0))) {
        if (state.stitching_initialized) {
            state.UpdatePanoramaWithAdjustments();
        }
    }
    
    // Auto-update on change
    if ((ab_changed || bc_changed) && state.auto_update_enabled && state.stitching_initialized) {
        state.UpdatePanoramaWithAdjustments();
    }
    
    if (!state.manual_adjustments_enabled) {
        ImGui::EndDisabled();
    }
    
    ImGui::End();
}

void DrawRTSPPanel(ApplicationState& state) {
    ImGui::Begin("RTSP Configuration");

    ImGui::Checkbox("Use RTSP Streams Instead of Static Images", &state.use_rtsp_streams);

    if (state.use_rtsp_streams) {
        for (int i = 0; i < 3; ++i) {
            ImGui::PushID(i);
            ImGui::InputText(state.camera_names[i], state.rtsp_urls[i], sizeof(state.rtsp_urls[i]));
            ImGui::PopID();
        }
        if (ImGui::Button("Reconnect Streams")) {
            // Cerrar previos si abiertos
            for (int i = 0; i < 3; ++i) {
                if (state.rtsp_caps[i].isOpened())
                    state.rtsp_caps[i].release();
            }
            state.CaptureRTSPFrames();
        }
    }

    ImGui::End();
}
