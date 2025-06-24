#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <GL/gl3w.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

// Camera calibration data structure
struct CameraIntrinsics {
    glm::mat3 camera_matrix;
    std::vector<float> distortion_coeffs;
    int image_width, image_height;
};

struct CameraExtrinsics {
    glm::vec3 rvec, tvec;
    glm::mat4 transformation_matrix;
};

struct CameraCalibration {
    std::string name;
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
    
    // Scaled versions for different output resolutions
    std::unordered_map<std::string, CameraIntrinsics> scaled_intrinsics;
};

// World coordinate system configuration
struct WorldCoordinateSystem {
    glm::vec2 bounds_min = glm::vec2(-4.0f, -2.5f);  // 8m × 5m test area
    glm::vec2 bounds_max = glm::vec2(4.0f, 2.5f);
    float z_plane = 0.0f;  // Z=0 plane as test bench surface
    glm::vec2 origin = glm::vec2(0.0f, 0.0f);  // Center of bench
    
    // Output image parameters
    int output_width = 1920;
    int output_height = 1080;
    float pixel_to_meter_ratio = 100.0f;  // pixels per meter
    
    // Cylindrical projection parameters
    enum ProjectionType {
        PLANAR = 0,
        CYLINDRICAL = 1,
        SPHERICAL = 2  // Future extension
    } projection_type = CYLINDRICAL;
    
    float cylinder_radius = 6.0f;  // Radius for cylindrical projection
    bool auto_cylinder_radius = true;  // Automatically calculate based on scene
};

// Overlap and blending configuration
struct BlendingConfig {
    float feather_size = 0.1f;  // Feathering size in meters
    float overlap_threshold = 0.1f;  // Minimum overlap for blending
    enum BlendMode {
        LINEAR = 0,
        MULTIBAND = 1,
        FEATHER = 2,
        NONE = 3
    } blend_mode = FEATHER;
};

// GPU resources for rendering pipeline
struct GPUResources {
    // Framebuffers for different resolutions
    std::unordered_map<std::string, GLuint> framebuffers;
    std::unordered_map<std::string, GLuint> color_textures;
    std::unordered_map<std::string, GLuint> depth_textures;
    
    // Camera input textures
    std::vector<GLuint> camera_textures;
    
    // Blend weight textures
    std::vector<GLuint> weight_textures;
    
    // Vertex data for world plane tessellation
    GLuint world_plane_vao, world_plane_vbo, world_plane_ebo;
    
    // Shader programs
    GLuint reprojection_shader;
    GLuint blend_shader;
    
    // Uniform buffer objects
    GLuint camera_matrices_ubo;
    GLuint blend_params_ubo;
};

// Camera viewing frustum on Z=0 plane
struct CameraFrustum {
    std::vector<glm::vec2> corners;  // Frustum corners on Z=0
    glm::vec2 center;
    float coverage_radius;
};

// Overlap region between cameras
struct OverlapRegion {
    std::vector<glm::vec2> boundary;
    float area;
    std::pair<int, int> camera_indices;  // Which cameras overlap
};

class StitchingPipeline {
public:
    StitchingPipeline();
    ~StitchingPipeline();
    
    // Phase 1: Data Structures & Initialization
    bool LoadCalibrationData(const std::string& json_path);
    bool LoadTestImages(const std::vector<std::string>& image_paths);
    void SetupWorldCoordinateSystem(const WorldCoordinateSystem& world_config);
    void AnalyzeCameraGeometry();
    
    // Phase 2: OpenGL Pipeline Architecture
    bool InitializeGPUResources();
    bool CreateFramebuffers(const std::vector<std::pair<int, int>>& resolutions);
    bool LoadShaders();
    void SetupTessellatedWorldPlane(int subdivisions_x = 200, int subdivisions_y = 120);
    
    // Phase 3: Mathematical Foundation & Rendering
    void UpdateCameraMatrices();
    void ComputeBlendWeights();
    void RenderStitchedOutput(const std::string& resolution = "1080p");
    
    // Coordinate transformations (now supports cylindrical projection)
    glm::vec3 WorldToCamera(const glm::vec3& world_point, size_t camera_index) const;
    glm::vec2 CameraToPixel(const glm::vec3& camera_point, size_t camera_index, 
                           const std::string& resolution = "original") const;
    glm::vec2 WorldToPixel(const glm::vec2& world_point, size_t camera_index, 
                          const std::string& resolution = "original") const;
    
    // Distortion handling
    glm::vec2 UndistortPoint(const glm::vec2& distorted_point, size_t camera_index) const;
    glm::vec2 DistortPoint(const glm::vec2& undistorted_point, size_t camera_index) const;
    
    // Cylindrical projection specific methods
    float GetCylinderRadius() const { 
        if (world_config_.auto_cylinder_radius) {
            float world_width = world_config_.bounds_max.x - world_config_.bounds_min.x;
            return world_width * 0.75f;
        }
        return world_config_.cylinder_radius; 
    }
    
    void SetCylinderRadius(float radius) { 
        world_config_.cylinder_radius = radius; 
        world_config_.auto_cylinder_radius = false;
    }
    
    // Getters
    GLuint GetOutputTexture(const std::string& resolution = "1080p") const;
    const std::vector<CameraCalibration>& GetCameras() const { return cameras_; }
    const WorldCoordinateSystem& GetWorldConfig() const { return world_config_; }
    const BlendingConfig& GetBlendingConfig() const { return blend_config_; }
    
    // Setters
    void SetBlendingConfig(const BlendingConfig& config) { blend_config_ = config; }
    void SetWorldConfig(const WorldCoordinateSystem& config) { world_config_ = config; }
    
    // Debug and visualization
    void DrawCameraFrustums() const;
    void DrawOverlapRegions() const;
    std::vector<float> GetCoverageMap() const;

private:
    // Core data
    std::vector<CameraCalibration> cameras_;
    WorldCoordinateSystem world_config_;
    BlendingConfig blend_config_;
    
    // GPU resources
    GPUResources gpu_resources_;
    
    // Geometry analysis
    std::vector<CameraFrustum> camera_frustums_;
    std::vector<OverlapRegion> overlap_regions_;
    
    // Test images (for initial implementation)
    std::vector<cv::Mat> test_images_;
    
    // Helper methods
    bool ParseCalibrationJSON(const json& calibration_json);
    void CreateScaledIntrinsics();
    glm::mat3 JsonToMat3(const json& json_matrix) const;
    glm::mat4 JsonToMat4(const json& json_matrix) const;
    glm::vec3 JsonToVec3(const json& json_vector) const;
    
    // Shader compilation helpers
    GLuint CompileShader(const std::string& source, GLenum type);
    GLuint CreateShaderProgram(const std::string& vertex_source, 
                              const std::string& fragment_source);
    
    // Geometry computation
    void ComputeCameraFrustum(size_t camera_index);
    void ComputeOverlapRegions();
    float ComputeBlendWeight(const glm::vec2& world_point, size_t camera_index) const;
    
    // GPU upload helpers
    void UploadCameraTextures();
    void UploadCameraMatrices();
    void UploadBlendParameters();
    
    // Validation
    bool ValidateCalibration() const;
    bool ValidateGPUResources() const;
};

// Utility functions
namespace StitchingUtils {
    // Matrix utilities
    glm::mat3 OpenCVToGLM(const cv::Mat& cv_matrix);
    cv::Mat GLMToOpenCV(const glm::mat3& glm_matrix);
    
    // Coordinate system utilities
    glm::vec2 WorldToOutputImage(const glm::vec2& world_point, 
                                const WorldCoordinateSystem& world_config);
    glm::vec2 OutputImageToWorld(const glm::vec2& image_point, 
                                const WorldCoordinateSystem& world_config);
    
    // Cylindrical projection utilities
    glm::vec3 ParametricToCylindrical(const glm::vec2& parametric_coords, float radius);
    glm::vec2 CylindricalToParametric(const glm::vec3& cylindrical_coords, float radius);
    
    // Geometric utilities
    bool PointInPolygon(const glm::vec2& point, const std::vector<glm::vec2>& polygon);
    float DistanceToPolygon(const glm::vec2& point, const std::vector<glm::vec2>& polygon);
    
    // Blending utilities
    float LinearBlend(float distance, float feather_size);
    float CosineBlend(float distance, float feather_size);
    float GaussianBlend(float distance, float feather_size);
}