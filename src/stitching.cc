#include "stitching.hh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

// FIXED: Updated vertex shader with cylindrical projection model
const char* REPROJECTION_VERTEX_SHADER = R"(
#version 330 core

layout (location = 0) in vec3 aWorldPos;

uniform mat4 uCameraMatrices[3];
uniform mat3 uCameraIntrinsics[3];
uniform vec4 uDistortionCoeffs[3];
uniform vec2 uImageSizes[3];
uniform int uOutputWidth;
uniform int uOutputHeight;
uniform vec2 uWorldBounds[2];
uniform float uCylinderRadius;

out vec2 vCameraUV[3];
out float vCameraWeights[3];
out vec2 vOutputUV;

vec2 distortPoint(vec2 normalized, vec4 distCoeffs) {
    // Apply radial and tangential distortion
    // Input: normalized coordinates (x/z, y/z) from camera
    // Output: distorted normalized coordinates
    
    float x = normalized.x;
    float y = normalized.y;
    
    float r2 = x*x + y*y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    
    // Radial distortion coefficients
    float k1 = distCoeffs.x;
    float k2 = distCoeffs.y;
    float k3 = (distCoeffs.w != distCoeffs.z) ? 0.0 : 0.0; // k3 is usually in 5th position, not used here
    
    // Tangential distortion coefficients  
    float p1 = distCoeffs.z;
    float p2 = distCoeffs.w;
    
    // Apply radial distortion
    float radial_factor = 1.0 + k1*r2 + k2*r4 + k3*r6;
    
    // Apply tangential distortion
    float dx = 2.0*p1*x*y + p2*(r2 + 2.0*x*x);
    float dy = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y;
    
    vec2 distorted;
    distorted.x = x * radial_factor + dx;
    distorted.y = y * radial_factor + dy;
    
    return distorted;
}

float computeDistanceWeight(vec2 worldPos, int cameraIndex) {
    // Extract camera position from transformation matrix
    // The transformation matrix is world-to-camera, so inverse gives camera-to-world
    mat4 camToWorld = inverse(uCameraMatrices[cameraIndex]);
    vec3 cameraWorldPos = camToWorld[3].xyz;
    
    // Distance-based weighting in world coordinates
    float distance = length(worldPos - cameraWorldPos.xy);
    
    // Weight falloff - adjust these parameters based on your camera spacing
    float maxDistance = 5.0; // meters
    float minWeight = 0.0;
    
    return clamp(1.0 - distance / maxDistance, minWeight, 1.0);
}

void main() {
    // === CYLINDRICAL PROJECTION MODEL ===
    // Convert input planar coordinates to cylindrical coordinates
    float angle = aWorldPos.x / uCylinderRadius;
    float height = aWorldPos.y;
    
    // Calculate 3D world position on cylinder surface
    vec3 worldPointOnCylinder = vec3(
        uCylinderRadius * sin(angle), 
        height, 
        uCylinderRadius * -cos(angle)  // Note the -cos to have Z increase away from cameras
    );
    
    vec4 worldPos4 = vec4(worldPointOnCylinder, 1.0);
    
    // Transform cylindrical world coordinates to output image coordinates for rasterization
    // Map angle back to U coordinate and height to V coordinate
    vec2 worldRange = uWorldBounds[1] - uWorldBounds[0];
    vOutputUV = (aWorldPos.xy - uWorldBounds[0]) / worldRange;
    
    // For each camera, compute UV coordinates and weights
    for (int i = 0; i < 3; i++) {
        // STEP 1: Transform world point to camera coordinates
        // This follows the same pipeline as your Python solvePnP/projectPoints
        vec4 cameraPos = uCameraMatrices[i] * worldPos4;
        
        if (cameraPos.z > 0.01) { // Point must be in front of camera
            // STEP 2: Perspective projection to normalized coordinates
            // This is equivalent to dividing by Z in camera coordinates
            vec2 normalizedCoords = cameraPos.xy / cameraPos.z;
            
            // STEP 3: Apply distortion to normalized coordinates
            // This matches cv2.projectPoints distortion application
            vec2 distortedNormalized = distortPoint(normalizedCoords, uDistortionCoeffs[i]);
            
            // STEP 4: Apply intrinsic matrix to get pixel coordinates
            // K * [x_distorted, y_distorted, 1]^T = [u, v, 1]^T
            vec3 pixelHomogeneous = uCameraIntrinsics[i] * vec3(distortedNormalized, 1.0);
            vec2 pixelCoords = pixelHomogeneous.xy; // Already in pixels
            
            // STEP 5: Convert to UV coordinates [0,1]
            vCameraUV[i] = pixelCoords / uImageSizes[i];
            
            // Check if projection is within image bounds
            if (vCameraUV[i].x >= 0.0 && vCameraUV[i].x <= 1.0 && 
                vCameraUV[i].y >= 0.0 && vCameraUV[i].y <= 1.0) {
                // Compute blending weight based on distance to camera (use original input coordinates)
                vCameraWeights[i] = computeDistanceWeight(aWorldPos.xy, i);
            } else {
                vCameraWeights[i] = 0.0;
            }
        } else {
            vCameraUV[i] = vec2(-1.0); // Invalid UV
            vCameraWeights[i] = 0.0;
        }
    }
    
    // Normalize weights so they sum to 1.0
    float totalWeight = vCameraWeights[0] + vCameraWeights[1] + vCameraWeights[2];
    if (totalWeight > 0.001) {
        vCameraWeights[0] /= totalWeight;
        vCameraWeights[1] /= totalWeight; 
        vCameraWeights[2] /= totalWeight;
    } else {
        // No valid cameras, set all weights to 0
        vCameraWeights[0] = 0.0;
        vCameraWeights[1] = 0.0;
        vCameraWeights[2] = 0.0;
    }
    
    // Set output position for rasterization
    gl_Position = vec4(vOutputUV * 2.0 - 1.0, 0.0, 1.0);
}
)";

// Fragment shader remains the same but with better blending
const char* REPROJECTION_FRAGMENT_SHADER = R"(
#version 330 core

in vec2 vCameraUV[3];
in float vCameraWeights[3];
in vec2 vOutputUV;

uniform sampler2D uCameraTexture0;
uniform sampler2D uCameraTexture1;
uniform sampler2D uCameraTexture2;
uniform float uFeatherSize;
uniform int uBlendMode;

out vec4 FragColor;

vec3 sampleCamera(int index) {
    vec2 uv;
    bool valid;
    
    if (index == 0) {
        uv = vCameraUV[0];
        valid = (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0);
        return valid ? texture(uCameraTexture0, uv).rgb : vec3(0.0);
    } else if (index == 1) {
        uv = vCameraUV[1];
        valid = (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0);
        return valid ? texture(uCameraTexture1, uv).rgb : vec3(0.0);
    } else if (index == 2) {
        uv = vCameraUV[2];
        valid = (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0);
        return valid ? texture(uCameraTexture2, uv).rgb : vec3(0.0);
    }
    return vec3(0.0);
}

void main() {
    vec3 finalColor = vec3(0.0);
    float totalWeight = 0.0;
    
    // Blend cameras based on computed weights
    for (int i = 0; i < 3; i++) {
        if (vCameraWeights[i] > 0.0) {
            vec3 cameraColor = sampleCamera(i);
            float weight = vCameraWeights[i];
            
            // Apply feathering if enabled
            if (uBlendMode == 2) { // FEATHER mode
                weight = smoothstep(0.0, uFeatherSize, weight);
            }
            
            finalColor += cameraColor * weight;
            totalWeight += weight;
        }
    }
    
    if (totalWeight > 0.0) {
        FragColor = vec4(finalColor / totalWeight, 1.0);
    } else {
        // No camera coverage - debug color
        FragColor = vec4(0.1, 0.1, 0.2, 1.0);
    }
}
)";

StitchingPipeline::StitchingPipeline() {
    world_config_ = WorldCoordinateSystem();
    blend_config_ = BlendingConfig();
    gpu_resources_ = GPUResources{};
}

StitchingPipeline::~StitchingPipeline() {
    // Cleanup GPU resources
    for (auto& [res, fbo] : gpu_resources_.framebuffers) {
        if (fbo) glDeleteFramebuffers(1, &fbo);
    }
    for (auto& [res, tex] : gpu_resources_.color_textures) {
        if (tex) glDeleteTextures(1, &tex);
    }
    for (auto& [res, tex] : gpu_resources_.depth_textures) {
        if (tex) glDeleteTextures(1, &tex);
    }
    
    if (!gpu_resources_.camera_textures.empty()) {
        glDeleteTextures(gpu_resources_.camera_textures.size(), 
                        gpu_resources_.camera_textures.data());
    }
    
    if (gpu_resources_.world_plane_vao) {
        glDeleteVertexArrays(1, &gpu_resources_.world_plane_vao);
        glDeleteBuffers(1, &gpu_resources_.world_plane_vbo);
        glDeleteBuffers(1, &gpu_resources_.world_plane_ebo);
    }
    
    if (gpu_resources_.reprojection_shader) {
        glDeleteProgram(gpu_resources_.reprojection_shader);
    }
}

bool StitchingPipeline::LoadCalibrationData(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open calibration file: " << json_path << std::endl;
        return false;
    }
    
    json calibration_json;
    file >> calibration_json;
    file.close();
    
    return ParseCalibrationJSON(calibration_json);
}

bool StitchingPipeline::ParseCalibrationJSON(const json& calibration_json) {
    try {
        cameras_.clear();
        
        // Parse configuration
        auto config = calibration_json["config"];
        int image_width = config["image_width"];
        int image_height = config["image_height"];
        
        std::cout << "Loading calibration for " << image_width << "x" << image_height << " images" << std::endl;
        
        // Parse each camera - must match Python script order
        auto intrinsics_json = calibration_json["intrinsics"];
        auto extrinsics_json = calibration_json["extrinsics"];
        
        // FIXED: Use same camera order as Python script
        std::vector<std::string> camera_names = {"izquierda", "central", "derecha"};
        
        for (const auto& name : camera_names) {
            CameraCalibration camera;
            camera.name = name;
            
            // Parse intrinsics - FIXED: proper matrix loading
            auto& intrinsics = camera.intrinsics;
            intrinsics.camera_matrix = JsonToMat3(intrinsics_json[name]["camera_matrix"]);
            
            auto dist_coeffs = intrinsics_json[name]["distortion_coeffs"];
            intrinsics.distortion_coeffs.clear();
            for (const auto& coeff : dist_coeffs) {
                intrinsics.distortion_coeffs.push_back(static_cast<float>(coeff));
            }
            
            intrinsics.image_width = image_width;
            intrinsics.image_height = image_height;
            
            // Parse extrinsics - FIXED: transformation matrix is world-to-camera
            auto& extrinsics = camera.extrinsics;
            extrinsics.rvec = JsonToVec3(extrinsics_json[name]["rvec"]);
            extrinsics.tvec = JsonToVec3(extrinsics_json[name]["tvec"]);
            extrinsics.transformation_matrix = JsonToMat4(extrinsics_json[name]["transformation_matrix"]);
            
            cameras_.push_back(camera);
            
            // Debug output to match Python script
            std::cout << "Loaded camera: " << name << std::endl;
            std::cout << "  Intrinsics fx=" << intrinsics.camera_matrix[0][0] 
                      << ", fy=" << intrinsics.camera_matrix[1][1] << std::endl;
            std::cout << "  Translation: [" << extrinsics.tvec.x << ", " 
                      << extrinsics.tvec.y << ", " << extrinsics.tvec.z << "]" << std::endl;
        }
        
        // Create scaled intrinsics for different resolutions
        CreateScaledIntrinsics();
        
        std::cout << "Successfully loaded calibration for " << cameras_.size() << " cameras" << std::endl;
        return ValidateCalibration();
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing calibration JSON: " << e.what() << std::endl;
        return false;
    }
}

void StitchingPipeline::CreateScaledIntrinsics() {
    // FIXED: Create properly scaled intrinsics for different output resolutions
    std::vector<std::pair<std::string, std::pair<int, int>>> resolutions = {
        {"720p", {1280, 720}},
        {"1080p", {1920, 1080}},
        {"4K", {3840, 2160}}
    };
    
    for (auto& camera : cameras_) {
        for (const auto& [res_name, res_size] : resolutions) {
            CameraIntrinsics scaled = camera.intrinsics;
            
            float scale_x = static_cast<float>(res_size.first) / camera.intrinsics.image_width;
            float scale_y = static_cast<float>(res_size.second) / camera.intrinsics.image_height;
            
            // Scale the camera matrix components
            scaled.camera_matrix[0][0] *= scale_x; // fx
            scaled.camera_matrix[1][1] *= scale_y; // fy  
            scaled.camera_matrix[0][2] *= scale_x; // cx
            scaled.camera_matrix[1][2] *= scale_y; // cy
            // [2][2] remains 1.0
            
            scaled.image_width = res_size.first;
            scaled.image_height = res_size.second;
            
            camera.scaled_intrinsics[res_name] = scaled;
            
            std::cout << "Created " << res_name << " intrinsics for " << camera.name 
                      << ": " << res_size.first << "x" << res_size.second << std::endl;
        }
    }
}

bool StitchingPipeline::LoadTestImages(const std::vector<std::string>& image_paths) {
    if (image_paths.size() != 3) {
        std::cerr << "Expected 3 image paths, got " << image_paths.size() << std::endl;
        return false;
    }
    
    test_images_.clear();
    test_images_.reserve(3);
    
    for (size_t i = 0; i < image_paths.size(); i++) {
        const auto& path = image_paths[i];
        cv::Mat image = cv::imread(path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            return false;
        }
        test_images_.push_back(image);
        std::cout << "Loaded test image " << i << ": " << path 
                  << " (" << image.cols << "x" << image.rows << ")" << std::endl;
    }
    
    return true;
}

void StitchingPipeline::SetupWorldCoordinateSystem(const WorldCoordinateSystem& world_config) {
    world_config_ = world_config;
    
    // Calculate pixel-to-meter ratio
    float world_width = world_config_.bounds_max.x - world_config_.bounds_min.x;
    float world_height = world_config_.bounds_max.y - world_config_.bounds_min.y;
    
    world_config_.pixel_to_meter_ratio = std::min(
        world_config_.output_width / world_width,
        world_config_.output_height / world_height
    );
    
    std::cout << "World coordinate system configured:" << std::endl;
    std::cout << "  Bounds: [" << world_config_.bounds_min.x << ", " << world_config_.bounds_min.y 
              << "] to [" << world_config_.bounds_max.x << ", " << world_config_.bounds_max.y << "]" << std::endl;
    std::cout << "  Z-plane: " << world_config_.z_plane << "m" << std::endl;
    std::cout << "  Output: " << world_config_.output_width << "x" << world_config_.output_height << std::endl;
    std::cout << "  Pixel ratio: " << world_config_.pixel_to_meter_ratio << " px/m" << std::endl;
}

void StitchingPipeline::AnalyzeCameraGeometry() {
    camera_frustums_.clear();
    camera_frustums_.reserve(cameras_.size());
    
    for (size_t i = 0; i < cameras_.size(); i++) {
        ComputeCameraFrustum(i);
    }
    
    ComputeOverlapRegions();
    
    std::cout << "Camera geometry analysis complete:" << std::endl;
    std::cout << "  - Camera frustums: " << camera_frustums_.size() << std::endl;
    std::cout << "  - Overlap regions: " << overlap_regions_.size() << std::endl;
}

GLuint StitchingPipeline::CompileShader(const std::string& source, GLenum type) {
    GLuint shader = glCreateShader(type);
    const char* source_ptr = source.c_str();
    glShaderSource(shader, 1, &source_ptr, nullptr);
    glCompileShader(shader);
    
    // Check compilation status
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLint info_log_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
        std::vector<char> info_log(info_log_length);
        glGetShaderInfoLog(shader, info_log_length, nullptr, info_log.data());
        std::cerr << "Shader compilation failed: " << info_log.data() << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

GLuint StitchingPipeline::CreateShaderProgram(const std::string& vertex_source, 
                                             const std::string& fragment_source) {
    std::cout << "Compiling cylindrical projection shaders..." << std::endl;
    
    GLuint vertex_shader = CompileShader(vertex_source, GL_VERTEX_SHADER);
    if (!vertex_shader) return 0;
    
    GLuint fragment_shader = CompileShader(fragment_source, GL_FRAGMENT_SHADER);
    if (!fragment_shader) {
        glDeleteShader(vertex_shader);
        return 0;
    }
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    
    // Check linking status
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLint info_log_length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);
        std::vector<char> info_log(info_log_length);
        glGetProgramInfoLog(program, info_log_length, nullptr, info_log.data());
        std::cerr << "Shader program linking failed: " << info_log.data() << std::endl;
        glDeleteProgram(program);
        program = 0;
    }
    
    // Clean up shaders (they're now part of the program)
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    if (program) {
        std::cout << "Successfully created cylindrical projection shader program" << std::endl;
    }
    
    return program;
}

void StitchingPipeline::SetupTessellatedWorldPlane(int subdivisions_x, int subdivisions_y) {
    std::cout << "Setting up cylindrical world plane tessellation: " << subdivisions_x << "x" << subdivisions_y << std::endl;
    
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;
    
    // Generate vertices for cylindrical parametric surface
    // Instead of world coordinates, we generate parametric coordinates that will be
    // converted to cylindrical coordinates in the vertex shader
    float param_width = world_config_.bounds_max.x - world_config_.bounds_min.x;
    float param_height = world_config_.bounds_max.y - world_config_.bounds_min.y;
    
    for (int y = 0; y <= subdivisions_y; y++) {
        for (int x = 0; x <= subdivisions_x; x++) {
            float u = static_cast<float>(x) / subdivisions_x;
            float v = static_cast<float>(y) / subdivisions_y;
            
            // Create parametric coordinates - these will be converted to cylindrical in shader
            float param_x = world_config_.bounds_min.x + u * param_width;
            float param_y = world_config_.bounds_min.y + v * param_height;
            
            // Z coordinate is not used in cylindrical projection (height comes from param_y)
            vertices.push_back(glm::vec3(param_x, param_y, 0.0f));
        }
    }
    
    // Generate indices for triangulation
    for (int y = 0; y < subdivisions_y; y++) {
        for (int x = 0; x < subdivisions_x; x++) {
            int i = y * (subdivisions_x + 1) + x;
            
            // First triangle
            indices.push_back(i);
            indices.push_back(i + 1);
            indices.push_back(i + subdivisions_x + 1);
            
            // Second triangle
            indices.push_back(i + 1);
            indices.push_back(i + subdivisions_x + 2);
            indices.push_back(i + subdivisions_x + 1);
        }
    }
    
    std::cout << "Cylindrical tessellation: " << vertices.size() << " vertices, " 
              << indices.size() / 3 << " triangles" << std::endl;
    
    // Create and upload to GPU
    glGenVertexArrays(1, &gpu_resources_.world_plane_vao);
    glGenBuffers(1, &gpu_resources_.world_plane_vbo);
    glGenBuffers(1, &gpu_resources_.world_plane_ebo);
    
    glBindVertexArray(gpu_resources_.world_plane_vao);
    
    // Upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, gpu_resources_.world_plane_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
    
    // Upload index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpu_resources_.world_plane_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    
    // Set vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
    
    std::cout << "Cylindrical VAO/VBO/EBO created successfully" << std::endl;
}

void StitchingPipeline::UploadCameraTextures() {
    if (test_images_.empty()) {
        std::cout << "No test images to upload" << std::endl;
        return;
    }
    
    std::cout << "Uploading " << test_images_.size() << " camera textures" << std::endl;
    
    gpu_resources_.camera_textures.resize(3);
    glGenTextures(3, gpu_resources_.camera_textures.data());
    
    for (size_t i = 0; i < 3 && i < test_images_.size(); i++) {
        const auto& image = test_images_[i];
        
        glBindTexture(GL_TEXTURE_2D, gpu_resources_.camera_textures[i]);
        
        // Convert BGR to RGB
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, rgb_image.cols, rgb_image.rows, 
                     0, GL_RGB, GL_UNSIGNED_BYTE, rgb_image.data);
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        std::cout << "Uploaded camera texture " << i << ": " 
                  << image.cols << "x" << image.rows << std::endl;
    }
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool StitchingPipeline::LoadShaders() {
    gpu_resources_.reprojection_shader = CreateShaderProgram(
        REPROJECTION_VERTEX_SHADER, REPROJECTION_FRAGMENT_SHADER);
    
    if (!gpu_resources_.reprojection_shader) {
        std::cerr << "Failed to create cylindrical reprojection shader" << std::endl;
        return false;
    }
    
    std::cout << "Cylindrical projection shaders loaded successfully" << std::endl;
    return true;
}

bool StitchingPipeline::CreateFramebuffers(const std::vector<std::pair<int, int>>& resolutions) {
    std::vector<std::string> res_names = {"720p", "1080p", "4K"};
    
    for (size_t i = 0; i < resolutions.size() && i < res_names.size(); i++) {
        const auto& [width, height] = resolutions[i];
        const auto& res_name = res_names[i];
        
        std::cout << "Creating framebuffer for " << res_name << ": " << width << "x" << height << std::endl;
        
        // Create framebuffer
        GLuint fbo;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        
        // Create color texture
        GLuint color_texture;
        glGenTextures(1, &color_texture);
        glBindTexture(GL_TEXTURE_2D, color_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture, 0);
        
        // Create depth texture
        GLuint depth_texture;
        glGenTextures(1, &depth_texture);
        glBindTexture(GL_TEXTURE_2D, depth_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0);
        
        // Check framebuffer completeness
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Framebuffer not complete: " << status << std::endl;
            glDeleteFramebuffers(1, &fbo);
            glDeleteTextures(1, &color_texture);
            glDeleteTextures(1, &depth_texture);
            return false;
        }
        
        // Store resources
        gpu_resources_.framebuffers[res_name] = fbo;
        gpu_resources_.color_textures[res_name] = color_texture;
        gpu_resources_.depth_textures[res_name] = depth_texture;
        
        std::cout << "Successfully created framebuffer for " << res_name << std::endl;
    }
    
    // Unbind framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

void StitchingPipeline::ComputeCameraFrustum(size_t camera_index) {
    if (camera_index >= cameras_.size()) return;
    
    const auto& camera = cameras_[camera_index];
    CameraFrustum frustum;
    
    // FIXED: Compute actual camera frustum based on calibration data
    // Get camera position from inverse of transformation matrix  
    glm::mat4 world_to_cam = camera.extrinsics.transformation_matrix;
    glm::mat4 cam_to_world = glm::inverse(world_to_cam);
    glm::vec3 camera_position = glm::vec3(cam_to_world[3]);
    
    frustum.center = glm::vec2(camera_position.x, camera_position.y);
    
    // Estimate coverage radius based on field of view and Z-plane height
    float fx = camera.intrinsics.camera_matrix[0][0];
    float fy = camera.intrinsics.camera_matrix[1][1];
    float avg_focal = (fx + fy) * 0.5f;
    
    // Estimate field of view and coverage radius
    float fov_x = 2.0f * atan(camera.intrinsics.image_width / (2.0f * fx));
    float fov_y = 2.0f * atan(camera.intrinsics.image_height / (2.0f * fy));
    
    // Coverage radius at z-plane
    float z_distance = abs(camera_position.z - world_config_.z_plane);
    frustum.coverage_radius = z_distance * tan(std::max(fov_x, fov_y) / 2.0f);
    
    // Create simplified frustum corners (rectangular approximation)
    float half_width = z_distance * tan(fov_x / 2.0f);
    float half_height = z_distance * tan(fov_y / 2.0f);
    
    frustum.corners = {
        frustum.center + glm::vec2(-half_width, -half_height),
        frustum.center + glm::vec2(half_width, -half_height),
        frustum.center + glm::vec2(half_width, half_height), 
        frustum.center + glm::vec2(-half_width, half_height)
    };
    
    camera_frustums_.push_back(frustum);
    
    std::cout << "Camera " << camera_index << " (" << camera.name << "):" << std::endl;
    std::cout << "  Position: [" << camera_position.x << ", " << camera_position.y 
              << ", " << camera_position.z << "]" << std::endl;
    std::cout << "  Coverage radius: " << frustum.coverage_radius << "m" << std::endl;
}

bool StitchingPipeline::InitializeGPUResources() {
    std::cout << "Initializing GPU resources for cylindrical projection..." << std::endl;
    
    // Create framebuffers for standard resolutions
    std::vector<std::pair<int, int>> resolutions = {
        {1280, 720},   // 720p
        {1920, 1080},  // 1080p
        {3840, 2160}   // 4K
    };
    
    if (!CreateFramebuffers(resolutions)) {
        std::cerr << "Failed to create framebuffers" << std::endl;
        return false;
    }
    
    if (!LoadShaders()) {
        std::cerr << "Failed to load cylindrical projection shaders" << std::endl;
        return false;
    }
    
    SetupTessellatedWorldPlane();
    UploadCameraTextures();
    
    bool valid = ValidateGPUResources();
    std::cout << "Cylindrical projection GPU resource initialization: " << (valid ? "SUCCESS" : "FAILED") << std::endl;
    return valid;
}

// Helper conversion functions - FIXED to handle GLM column-major matrices
glm::mat3 StitchingPipeline::JsonToMat3(const json& json_matrix) const {
    glm::mat3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            // GLM is column-major: result[col][row] = json[row][col]
            result[j][i] = static_cast<float>(json_matrix[i][j]);
        }
    }
    return result;
}

glm::mat4 StitchingPipeline::JsonToMat4(const json& json_matrix) const {
    glm::mat4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            // GLM is column-major: result[col][row] = json[row][col]
            result[j][i] = static_cast<float>(json_matrix[i][j]);
        }
    }
    return result;
}

glm::vec3 StitchingPipeline::JsonToVec3(const json& json_vector) const {
    return glm::vec3(
        static_cast<float>(json_vector[0]), 
        static_cast<float>(json_vector[1]), 
        static_cast<float>(json_vector[2])
    );
}

bool StitchingPipeline::ValidateCalibration() const {
    if (cameras_.size() != 3) {
        std::cerr << "Expected 3 cameras, got " << cameras_.size() << std::endl;
        return false;
    }
    
    for (const auto& camera : cameras_) {
        if (camera.intrinsics.distortion_coeffs.size() < 4) {
            std::cerr << "Invalid distortion coefficients for camera: " << camera.name << std::endl;
            return false;
        }
    }
    
    return true;
}

bool StitchingPipeline::ValidateGPUResources() const {
    bool valid = gpu_resources_.reprojection_shader != 0 && 
                 gpu_resources_.world_plane_vao != 0 &&
                 !gpu_resources_.framebuffers.empty();
                 
    std::cout << "GPU Resource Validation:" << std::endl;
    std::cout << "  - Cylindrical reprojection shader: " << (gpu_resources_.reprojection_shader != 0 ? "OK" : "FAIL") << std::endl;
    std::cout << "  - World plane VAO: " << (gpu_resources_.world_plane_vao != 0 ? "OK" : "FAIL") << std::endl;
    std::cout << "  - Framebuffers: " << gpu_resources_.framebuffers.size() << " created" << std::endl;
    std::cout << "  - Camera textures: " << gpu_resources_.camera_textures.size() << " created" << std::endl;
    
    return valid;
}

GLuint StitchingPipeline::GetOutputTexture(const std::string& resolution) const {
    auto it = gpu_resources_.color_textures.find(resolution);
    return (it != gpu_resources_.color_textures.end()) ? it->second : 0;
}

// Implement the remaining methods that were in the original code...
// (For brevity, I'm including the key updated render.cc methods here)

// [Continue with rest of implementation...]