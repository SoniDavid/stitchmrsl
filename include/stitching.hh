#ifndef STITCHING_HH
#define STITCHING_HH

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// --- ENUMS ---
enum class BlendingMode {
    AVERAGE,
    FEATHERING,
    // MULTI_BAND // Future option
};


// --- DATA STRUCTURES ---

struct CameraIntrinsics {
    std::string model = "fisheye";
    cv::Mat camera_matrix;
    std::vector<double> distortion_coeffs;
    int image_width;
    int image_height;
};

struct PrecomputedTransform {
    bool valid = false;
    cv::Mat similarity; // 2x3 CV_64F matrix
    double mean_error = -1.0;
};

struct CameraCalibration {
    std::string name;
    CameraIntrinsics intrinsics;
};

// --- STITCHING PIPELINE CLASS ---

class StitchingPipeline {
public:
    StitchingPipeline();
    ~StitchingPipeline();

    // --- SETUP & LOADING ---
    bool LoadIntrinsicsData(const std::string& intrinsics_path);
    bool LoadExtrinsicsData(const std::string& extrinsics_path);
    bool LoadTestImages(const std::vector<std::string>& image_paths);
    bool LoadTestImagesFromMats(const std::vector<cv::Mat>& images);

    // --- CORE STITCHING LOGIC ---
    cv::Mat CreatePanoramaFromPrecomputed();
    cv::Mat CreatePanoramaWithCustomTransforms(const cv::Mat& custom_ab_transform, const cv::Mat& custom_bc_transform);
    void SetBlendingMode(BlendingMode mode);

private:
    // --- HELPER METHODS ---
    void PrecomputeRectificationMaps();
    cv::Mat RectifyImageFisheye(const cv::Mat& image, const std::string& name, const CameraIntrinsics& intrinsics);
    void BlendInPlace(cv::Mat& base, const cv::Mat& overlay, BlendingMode mode);


    // --- MEMBER VARIABLES ---
    BlendingMode blending_mode_ = BlendingMode::FEATHERING;
    std::vector<CameraCalibration> cameras_;
    std::vector<cv::Mat> test_images_;
    std::vector<cv::Mat> rectified_images_;

    // Pre-computed transformations loaded from JSON
    PrecomputedTransform loaded_ab_transform_; // izquierda -> central
    PrecomputedTransform loaded_bc_transform_; // central -> derecha

    // Rectification Maps
    std::map<std::string, cv::Mat> map1_;
    std::map<std::string, cv::Mat> map2_;
    bool maps_ready_ = false;
};

#endif // STITCHING_HH