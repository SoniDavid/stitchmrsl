#include "gui.hh"
#include <iostream>

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

    std::vector<std::string> paths;
    for(int i=0; i<3; ++i) paths.push_back(test_image_paths[i]);
    test_images_loaded = pipeline->LoadTestImages(paths);

    if (calibration_loaded && test_images_loaded) {
        stitching_initialized = true;
        status_message = "Ready. Click 'Create Panorama'.";
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
    ImGui::Text("Viewer Options");
    if (ImGui::SliderFloat("Rotation", &state.rotation_degrees, -180.0f, 180.0f, "%.1f deg")) {
        // Snap to right angles
        const float snap_threshold = 2.0f; // Degrees
        const float angles[] = {-180.0f, -90.0f, 0.0f, 90.0f, 180.0f};
        for (float angle : angles) {
            if (fabs(state.rotation_degrees - angle) < snap_threshold) {
                state.rotation_degrees = angle;
                break;
            }
        }
    }

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