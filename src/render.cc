#include <GL/gl3w.h>
#include "stitching.hh"

void StitchingPipeline::UpdateCameraMatrices() {
    if (!gpu_resources_.reprojection_shader) return;
    
    glUseProgram(gpu_resources_.reprojection_shader);
    
    std::cout << "Uploading camera matrices and cylindrical parameters to GPU..." << std::endl;
    
    // === UPLOAD CYLINDER RADIUS UNIFORM (NEW) ===
    GLint cylinder_location = glGetUniformLocation(gpu_resources_.reprojection_shader, "uCylinderRadius");
    if (cylinder_location >= 0) {
        // Calculate optimal cylinder radius based on camera spacing and scene geometry
        float world_width = world_config_.bounds_max.x - world_config_.bounds_min.x;
        
        // Use a radius that corresponds to roughly 1.5x the world width for good cylindrical coverage
        // This ensures the cylinder wraps around the scene nicely
        float cylinder_radius = world_width * 0.75f; // Adjust this value as needed
        
        // Alternative: Use pixel-based radius for better correlation with intrinsics
        // float cylinder_radius = 1500.0f; // In world units, roughly matching focal length scale
        
        glUniform1f(cylinder_location, cylinder_radius);
        std::cout << "  Cylinder radius set to: " << cylinder_radius << " world units" << std::endl;
    } else {
        std::cerr << "Failed to find uniform: uCylinderRadius" << std::endl;
    }
    
    // Upload camera transformation matrices (world-to-camera transforms)
    for (size_t i = 0; i < cameras_.size() && i < 3; i++) {
        // FIXED: Upload transformation matrix correctly
        std::string uniform_name = "uCameraMatrices[" + std::to_string(i) + "]";
        GLint location = glGetUniformLocation(gpu_resources_.reprojection_shader, uniform_name.c_str());
        if (location >= 0) {
            // GLM matrices are already column-major, OpenGL expects column-major
            // glUniformMatrix4fv(location, 1, GL_FALSE, &cameras_[i].extrinsics.transformation_matrix[0][0]);
            glm::mat4 camera_to_world = cameras_[i].extrinsics.transformation_matrix;
            glm::mat4 world_to_camera = glm::inverse(camera_to_world); // Invert the matrix!
            glUniformMatrix4fv(location, 1, GL_FALSE, &world_to_camera[0][0]);
            
            std::cout << "  Camera " << i << " (" << cameras_[i].name << ") transformation matrix uploaded" << std::endl;
        } else {
            std::cerr << "Failed to find uniform: " << uniform_name << std::endl;
        }
        
        // FIXED: Upload intrinsic matrices correctly
        uniform_name = "uCameraIntrinsics[" + std::to_string(i) + "]";
        location = glGetUniformLocation(gpu_resources_.reprojection_shader, uniform_name.c_str());
        if (location >= 0) {
            // Use current resolution intrinsics if available
            const CameraIntrinsics* intrinsics = &cameras_[i].intrinsics;
            
            // Try to find scaled intrinsics for current output
            std::string current_res = "1080p"; // Default
            if (world_config_.output_width == 1280) current_res = "720p";
            else if (world_config_.output_width == 3840) current_res = "4K";
            
            auto scaled_it = cameras_[i].scaled_intrinsics.find(current_res);
            if (scaled_it != cameras_[i].scaled_intrinsics.end()) {
                intrinsics = &scaled_it->second;
                std::cout << "  Using " << current_res << " intrinsics for camera " << i << std::endl;
            }
            
            glUniformMatrix3fv(location, 1, GL_FALSE, &intrinsics->camera_matrix[0][0]);
        }
        
        // FIXED: Upload distortion coefficients (ensure we have at least 4)
        uniform_name = "uDistortionCoeffs[" + std::to_string(i) + "]";
        location = glGetUniformLocation(gpu_resources_.reprojection_shader, uniform_name.c_str());
        if (location >= 0) {
            glm::vec4 distortion(0.0f); // Initialize to zero
            
            const auto& dist_coeffs = cameras_[i].intrinsics.distortion_coeffs;
            if (dist_coeffs.size() >= 2) {
                distortion.x = dist_coeffs[0]; // k1
                distortion.y = dist_coeffs[1]; // k2
            }
            if (dist_coeffs.size() >= 4) {
                distortion.z = dist_coeffs[2]; // p1
                distortion.w = dist_coeffs[3]; // p2
            }
            // Note: k3 (5th coefficient) not used in this implementation
            
            glUniform4fv(location, 1, &distortion[0]);
            
            std::cout << "  Camera " << i << " distortion: k1=" << distortion.x 
                      << ", k2=" << distortion.y << ", p1=" << distortion.z 
                      << ", p2=" << distortion.w << std::endl;
        }
        
        // FIXED: Upload image sizes
        uniform_name = "uImageSizes[" + std::to_string(i) + "]";
        location = glGetUniformLocation(gpu_resources_.reprojection_shader, uniform_name.c_str());
        if (location >= 0) {
            glm::vec2 imageSize(
                static_cast<float>(cameras_[i].intrinsics.image_width), 
                static_cast<float>(cameras_[i].intrinsics.image_height)
            );
            glUniform2fv(location, 1, &imageSize[0]);
        }
    }
    
    // FIXED: Upload world bounds
    GLint location = glGetUniformLocation(gpu_resources_.reprojection_shader, "uWorldBounds[0]");
    if (location >= 0) {
        glUniform2fv(location, 1, &world_config_.bounds_min[0]);
    }
    location = glGetUniformLocation(gpu_resources_.reprojection_shader, "uWorldBounds[1]");
    if (location >= 0) {
        glUniform2fv(location, 1, &world_config_.bounds_max[0]);
    }
    
    // Upload output dimensions
    location = glGetUniformLocation(gpu_resources_.reprojection_shader, "uOutputWidth");
    if (location >= 0) {
        glUniform1i(location, world_config_.output_width);
    }
    location = glGetUniformLocation(gpu_resources_.reprojection_shader, "uOutputHeight");
    if (location >= 0) {
        glUniform1i(location, world_config_.output_height);
    }
    
    std::cout << "Camera matrix and cylindrical parameter upload complete" << std::endl;
}

void StitchingPipeline::ComputeBlendWeights() {
    // For now, rely on shader-based distance weighting
    // Future: implement texture-based blend weight maps
    std::cout << "Using shader-based blend weight computation for cylindrical projection" << std::endl;
}

void StitchingPipeline::RenderStitchedOutput(const std::string& resolution) {
    if (!ValidateGPUResources()) {
        std::cerr << "GPU resources not properly initialized" << std::endl;
        return;
    }
    
    // Find framebuffer for requested resolution
    auto fbo_it = gpu_resources_.framebuffers.find(resolution);
    if (fbo_it == gpu_resources_.framebuffers.end()) {
        std::cerr << "Framebuffer not found for resolution: " << resolution << std::endl;
        return;
    }
    
    std::cout << "Rendering cylindrical stitched output at " << resolution << "..." << std::endl;
    
    // FIXED: Bind the target framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_it->second);
    
    // Set viewport based on resolution
    int width, height;
    if (resolution == "720p") { 
        width = 1280; height = 720; 
    } else if (resolution == "1080p") { 
        width = 1920; height = 1080; 
    } else if (resolution == "4K") { 
        width = 3840; height = 2160; 
    } else { 
        width = world_config_.output_width; 
        height = world_config_.output_height; 
    }
    
    glViewport(0, 0, width, height);
    std::cout << "  Viewport: " << width << "x" << height << std::endl;
    
    // Clear the framebuffer
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Use the reprojection shader
    glUseProgram(gpu_resources_.reprojection_shader);
    
    // Update camera matrices and parameters (including cylinder radius)
    UpdateCameraMatrices();
    
    // FIXED: Bind camera textures to separate texture units
    std::vector<std::string> texture_uniforms = {"uCameraTexture0", "uCameraTexture1", "uCameraTexture2"};
    
    for (size_t i = 0; i < gpu_resources_.camera_textures.size() && i < 3; i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, gpu_resources_.camera_textures[i]);
        
        GLint location = glGetUniformLocation(gpu_resources_.reprojection_shader, texture_uniforms[i].c_str());
        if (location >= 0) {
            glUniform1i(location, i);
            std::cout << "  Bound camera texture " << i << " to unit " << i << std::endl;
        }
    }
    
    // FIXED: Upload blend parameters
    GLint location = glGetUniformLocation(gpu_resources_.reprojection_shader, "uFeatherSize");
    if (location >= 0) {
        glUniform1f(location, blend_config_.feather_size);
    }
    
    location = glGetUniformLocation(gpu_resources_.reprojection_shader, "uBlendMode");
    if (location >= 0) {
        glUniform1i(location, static_cast<int>(blend_config_.blend_mode));
    }
    
    // FIXED: Render the tessellated cylindrical world plane
    glBindVertexArray(gpu_resources_.world_plane_vao);
    
    // Get number of indices to draw
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpu_resources_.world_plane_ebo);
    GLint buffer_size;
    glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &buffer_size);
    int num_indices = buffer_size / sizeof(unsigned int);
    
    std::cout << "  Drawing " << num_indices / 3 << " triangles (cylindrical surface)" << std::endl;
    
    // Enable depth testing for proper rendering
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    
    // Draw the cylindrical world surface
    glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, 0);
    
    // Check for OpenGL errors
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error during cylindrical rendering: " << error << std::endl;
    }
    
    // Cleanup
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glActiveTexture(GL_TEXTURE0);
    
    std::cout << "Cylindrical stitched output rendering complete" << std::endl;
}

// FIXED: Coordinate transformation methods now supporting cylindrical projection
glm::vec3 StitchingPipeline::WorldToCamera(const glm::vec3& world_point, size_t camera_index) const {
    if (camera_index >= cameras_.size()) return glm::vec3(0.0f);
    
    // Apply world-to-camera transformation
    const auto& transform = cameras_[camera_index].extrinsics.transformation_matrix;
    glm::vec4 world_point4(world_point, 1.0f);
    glm::vec4 camera_point4 = transform * world_point4;
    
    return glm::vec3(camera_point4) / camera_point4.w;
}

glm::vec2 StitchingPipeline::CameraToPixel(const glm::vec3& camera_point, size_t camera_index, 
                                          const std::string& resolution) const {
    if (camera_index >= cameras_.size()) return glm::vec2(-1.0f);
    
    // Get the appropriate intrinsic matrix
    const CameraIntrinsics* intrinsics = &cameras_[camera_index].intrinsics;
    if (resolution != "original") {
        auto it = cameras_[camera_index].scaled_intrinsics.find(resolution);
        if (it != cameras_[camera_index].scaled_intrinsics.end()) {
            intrinsics = &it->second;
        }
    }
    
    // FIXED: Project to normalized device coordinates (matching Python cv2.projectPoints)
    if (camera_point.z <= 0.0f) return glm::vec2(-1.0f); // Behind camera
    
    // Perspective projection: divide by Z
    glm::vec2 normalized_coords = glm::vec2(camera_point.x / camera_point.z, camera_point.y / camera_point.z);
    
    // Apply distortion to normalized coordinates
    glm::vec2 distorted_normalized = DistortPoint(normalized_coords, camera_index);
    
    // Apply intrinsic matrix to get pixel coordinates
    glm::vec3 pixel_homogeneous = intrinsics->camera_matrix * glm::vec3(distorted_normalized, 1.0f);
    
    return glm::vec2(pixel_homogeneous.x, pixel_homogeneous.y);
}

glm::vec2 StitchingPipeline::WorldToPixel(const glm::vec2& world_point, size_t camera_index, 
                                         const std::string& resolution) const {
    // For cylindrical projection, we need to convert parametric coordinates to 3D cylindrical coordinates
    float world_width = world_config_.bounds_max.x - world_config_.bounds_min.x;
    float cylinder_radius = world_width * 0.75f; // Same as in shader
    
    // Convert parametric coordinates to cylindrical 3D coordinates
    float angle = world_point.x / cylinder_radius;
    float height = world_point.y;
    
    glm::vec3 world_point3(
        cylinder_radius * sin(angle),
        height,
        cylinder_radius * -cos(angle)
    );
    
    glm::vec3 camera_point = WorldToCamera(world_point3, camera_index);
    return CameraToPixel(camera_point, camera_index, resolution);
}

glm::vec2 StitchingPipeline::UndistortPoint(const glm::vec2& distorted_point, size_t camera_index) const {
    if (camera_index >= cameras_.size()) return distorted_point;
    
    const auto& dist_coeffs = cameras_[camera_index].intrinsics.distortion_coeffs;
    if (dist_coeffs.size() < 4) return distorted_point;
    
    // FIXED: Iterative undistortion (more accurate than single-pass)
    glm::vec2 undistorted = distorted_point;
    
    // Iterative refinement
    for (int iter = 0; iter < 5; iter++) {
        glm::vec2 distorted_estimate = DistortPoint(undistorted, camera_index);
        glm::vec2 error = distorted_point - distorted_estimate;
        undistorted += error;
    }
    
    return undistorted;
}

glm::vec2 StitchingPipeline::DistortPoint(const glm::vec2& undistorted_point, size_t camera_index) const {
    if (camera_index >= cameras_.size()) return undistorted_point;
    
    const auto& dist_coeffs = cameras_[camera_index].intrinsics.distortion_coeffs;
    if (dist_coeffs.size() < 4) return undistorted_point;
    
    // FIXED: Apply distortion model matching OpenCV cv2.projectPoints
    float x = undistorted_point.x;
    float y = undistorted_point.y;
    
    float r2 = x*x + y*y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    
    // Radial distortion
    float k1 = dist_coeffs[0];
    float k2 = dist_coeffs[1]; 
    float k3 = (dist_coeffs.size() > 4) ? dist_coeffs[4] : 0.0f;
    
    float radial_factor = 1.0f + k1*r2 + k2*r4 + k3*r6;
    
    // Tangential distortion
    float p1 = dist_coeffs[2];
    float p2 = dist_coeffs[3];
    
    glm::vec2 tangential(
        2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x),
        p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y
    );
    
    return undistorted_point * radial_factor + tangential;
}

void StitchingPipeline::ComputeOverlapRegions() {
    overlap_regions_.clear();
    
    // FIXED: Compute actual overlap regions based on camera frustums
    for (size_t i = 0; i < camera_frustums_.size(); i++) {
        for (size_t j = i + 1; j < camera_frustums_.size(); j++) {
            OverlapRegion overlap;
            overlap.camera_indices = {static_cast<int>(i), static_cast<int>(j)};
            
            // Simple overlap computation - intersection of coverage circles
            const auto& frustum1 = camera_frustums_[i];
            const auto& frustum2 = camera_frustums_[j];
            
            float distance = glm::length(frustum1.center - frustum2.center);
            float radius_sum = frustum1.coverage_radius + frustum2.coverage_radius;
            
            if (distance < radius_sum) {
                // Cameras have overlapping coverage
                overlap.area = 1.0f; // Simplified - could compute actual intersection area
                overlap_regions_.push_back(overlap);
                
                std::cout << "Overlap detected between cameras " << i << " and " << j 
                          << " (distance: " << distance << "m)" << std::endl;
            }
        }
    }
    
    std::cout << "Computed " << overlap_regions_.size() << " overlap regions for cylindrical projection" << std::endl;
}

float StitchingPipeline::ComputeBlendWeight(const glm::vec2& world_point, size_t camera_index) const {
    if (camera_index >= camera_frustums_.size()) return 0.0f;
    
    const auto& frustum = camera_frustums_[camera_index];
    float distance = glm::length(world_point - frustum.center);
    
    // Distance-based weighting with smooth falloff
    float normalized_distance = distance / frustum.coverage_radius;
    return std::max(0.0f, 1.0f - normalized_distance);
}

// Debug and visualization methods
void StitchingPipeline::DrawCameraFrustums() const {
    std::cout << "Camera Frustums (Cylindrical Projection):" << std::endl;
    for (size_t i = 0; i < camera_frustums_.size(); i++) {
        const auto& frustum = camera_frustums_[i];
        std::cout << "  Camera " << i << " (" << cameras_[i].name << "):" << std::endl;
        std::cout << "    Center: (" << frustum.center.x << ", " << frustum.center.y << ")" << std::endl;
        std::cout << "    Coverage radius: " << frustum.coverage_radius << "m" << std::endl;
    }
}

void StitchingPipeline::DrawOverlapRegions() const {
    std::cout << "Overlap Regions (Cylindrical Projection):" << std::endl;
    for (const auto& overlap : overlap_regions_) {
        std::cout << "  Cameras " << overlap.camera_indices.first 
                  << " and " << overlap.camera_indices.second 
                  << ", area: " << overlap.area << std::endl;
    }
}

std::vector<float> StitchingPipeline::GetCoverageMap() const {
    std::vector<float> coverage_map;
    
    int map_width = 200, map_height = 120;
    coverage_map.resize(map_width * map_height, 0.0f);
    
    float world_width = world_config_.bounds_max.x - world_config_.bounds_min.x;
    float world_height = world_config_.bounds_max.y - world_config_.bounds_min.y;
    
    std::cout << "Computing cylindrical coverage map (" << map_width << "x" << map_height << ")..." << std::endl;
    
    for (int y = 0; y < map_height; y++) {
        for (int x = 0; x < map_width; x++) {
            float u = static_cast<float>(x) / (map_width - 1);
            float v = static_cast<float>(y) / (map_height - 1);
            
            glm::vec2 world_point(
                world_config_.bounds_min.x + u * world_width,
                world_config_.bounds_min.y + v * world_height
            );
            
            // Compute total coverage from all cameras
            float total_weight = 0.0f;
            for (size_t cam = 0; cam < cameras_.size(); cam++) {
                total_weight += ComputeBlendWeight(world_point, cam);
            }
            
            coverage_map[y * map_width + x] = total_weight;
        }
    }
    
    return coverage_map;
}

// Utility namespace implementations matching Python coordinate transformations
namespace StitchingUtils {
    glm::mat3 OpenCVToGLM(const cv::Mat& cv_matrix) {
        glm::mat3 result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result[j][i] = cv_matrix.at<double>(i, j); // Column-major for GLM
            }
        }
        return result;
    }
    
    cv::Mat GLMToOpenCV(const glm::mat3& glm_matrix) {
        cv::Mat result = cv::Mat::zeros(3, 3, CV_64F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.at<double>(i, j) = glm_matrix[j][i]; // Row-major for OpenCV
            }
        }
        return result;
    }
    
    glm::vec2 WorldToOutputImage(const glm::vec2& world_point, 
                                const WorldCoordinateSystem& world_config) {
        glm::vec2 world_range = world_config.bounds_max - world_config.bounds_min;
        glm::vec2 normalized = (world_point - world_config.bounds_min) / world_range;
        
        return glm::vec2(
            normalized.x * world_config.output_width,
            normalized.y * world_config.output_height
        );
    }
    
    glm::vec2 OutputImageToWorld(const glm::vec2& image_point, 
                                const WorldCoordinateSystem& world_config) {
        glm::vec2 normalized(
            image_point.x / world_config.output_width,
            image_point.y / world_config.output_height
        );
        
        glm::vec2 world_range = world_config.bounds_max - world_config.bounds_min;
        return world_config.bounds_min + normalized * world_range;
    }
    
    bool PointInPolygon(const glm::vec2& point, const std::vector<glm::vec2>& polygon) {
        bool inside = false;
        int j = polygon.size() - 1;
        
        for (size_t i = 0; i < polygon.size(); i++) {
            if (((polygon[i].y > point.y) != (polygon[j].y > point.y)) &&
                (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / 
                           (polygon[j].y - polygon[i].y) + polygon[i].x)) {
                inside = !inside;
            }
            j = i;
        }
        
        return inside;
    }
    
    float DistanceToPolygon(const glm::vec2& point, const std::vector<glm::vec2>& polygon) {
        float min_distance = std::numeric_limits<float>::max();
        
        for (size_t i = 0; i < polygon.size(); i++) {
            size_t j = (i + 1) % polygon.size();
            
            glm::vec2 edge = polygon[j] - polygon[i];
            glm::vec2 to_point = point - polygon[i];
            
            float t = std::max(0.0f, std::min(1.0f, glm::dot(to_point, edge) / glm::dot(edge, edge)));
            glm::vec2 closest = polygon[i] + t * edge;
            
            float distance = glm::length(point - closest);
            min_distance = std::min(min_distance, distance);
        }
        
        return min_distance;
    }
    
    float LinearBlend(float distance, float feather_size) {
        if (feather_size <= 0.0f) return distance > 0.0f ? 0.0f : 1.0f;
        return std::max(0.0f, 1.0f - distance / feather_size);
    }
    
    float CosineBlend(float distance, float feather_size) {
        if (feather_size <= 0.0f) return distance > 0.0f ? 0.0f : 1.0f;
        if (distance >= feather_size) return 0.0f;
        
        float normalized = distance / feather_size;
        return 0.5f * (1.0f + std::cos(normalized * M_PI));
    }
    
    float GaussianBlend(float distance, float feather_size) {
        if (feather_size <= 0.0f) return distance > 0.0f ? 0.0f : 1.0f;
        
        float sigma = feather_size / 3.0f; // 3-sigma rule
        return std::exp(-(distance * distance) / (2.0f * sigma * sigma));
    }
}