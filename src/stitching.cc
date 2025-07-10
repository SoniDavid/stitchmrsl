#include "stitching.hh"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

StitchingPipeline::StitchingPipeline() {}
StitchingPipeline::~StitchingPipeline() {}

// --- PUBLIC METHODS ---

bool StitchingPipeline::LoadIntrinsicsData(const std::string& intrinsics_path) {
    std::ifstream file(intrinsics_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open intrinsics file: " << intrinsics_path << std::endl;
        return false;
    }

    json data;
    file >> data;
    auto intrinsics_json = data.value("cameras", json::object());
    if (intrinsics_json.empty()) {
         std::cerr << "ERROR: 'cameras' key not found in intrinsics JSON." << std::endl;
        return false;
    }

    cameras_.clear();
    std::vector<std::string> camera_names = {"izquierda", "central", "derecha"};

    try {
        for (const auto& name : camera_names) {
            if (!intrinsics_json.contains(name)) {
                 std::cerr << "ERROR: Calibration for camera '" << name << "' not found." << std::endl;
                return false;
            }
            CameraCalibration camera;
            camera.name = name;
            auto cam_data = intrinsics_json[name];

            // Load camera matrix (K)
            camera.intrinsics.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    camera.intrinsics.camera_matrix.at<double>(i, j) = cam_data["K"][i][j];
                }
            }

            // Load distortion coefficients
            camera.intrinsics.distortion_coeffs.clear();
            for (const auto& dist_row : cam_data["dist"]) {
                camera.intrinsics.distortion_coeffs.push_back(static_cast<double>(dist_row[0]));
            }

            auto img_size = cam_data["image_size"];
            camera.intrinsics.image_width = img_size[0];
            camera.intrinsics.image_height = img_size[1];
            camera.intrinsics.model = cam_data.value("model", "fisheye");
            cameras_.push_back(camera);
            std::cout << "Loaded intrinsics for camera: " << name << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing intrinsics JSON: " << e.what() << std::endl;
        return false;
    }
    PrecomputeRectificationMaps();
    return true;
}

bool StitchingPipeline::LoadExtrinsicsData(const std::string& extrinsics_path) {
    std::ifstream file(extrinsics_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open extrinsics file: " << extrinsics_path << std::endl;
        return false;
    }
    json extrinsics_json;
    file >> extrinsics_json;

    try {
        auto best_transforms_json = extrinsics_json["best_transforms"];
        auto results_json = extrinsics_json["results"];

        // Load AB transform (central -> izquierda)
        if (best_transforms_json.contains("AB_similarity") && results_json.contains("AB_similarity")) {
            cv::Mat ab_matrix_3x3(3, 3, CV_64F);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    ab_matrix_3x3.at<double>(i, j) = best_transforms_json["AB_similarity"][i][j];
                }
            }
            // Extract the 2x3 similarity part
            loaded_ab_transform_.similarity = ab_matrix_3x3(cv::Rect(0, 0, 3, 2)).clone();
            loaded_ab_transform_.mean_error = results_json["AB_similarity"]["mean_error"].get<double>();
            loaded_ab_transform_.valid = true;
            std::cout << "✓ Loaded AB similarity transform from best_transforms" << std::endl;
        }

        // Load BC transform (derecha -> central)
        if (best_transforms_json.contains("BC_similarity") && results_json.contains("BC_similarity")) {
            cv::Mat bc_matrix_3x3(3, 3, CV_64F);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    bc_matrix_3x3.at<double>(i, j) = best_transforms_json["BC_similarity"][i][j];
                }
            }
            // Extract the 2x3 similarity part
            loaded_bc_transform_.similarity = bc_matrix_3x3(cv::Rect(0, 0, 3, 2)).clone();
            loaded_bc_transform_.mean_error = results_json["BC_similarity"]["mean_error"].get<double>();
            loaded_bc_transform_.valid = true;
            std::cout << "✓ Loaded BC similarity transform from best_transforms" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error parsing extrinsics JSON: " << e.what() << std::endl;
        return false;
    }
    return loaded_ab_transform_.valid && loaded_bc_transform_.valid;
}

bool StitchingPipeline::LoadTestImages(const std::vector<std::string>& image_paths) {
    if (image_paths.size() != 3) {
        std::cerr << "Expected 3 image paths, got " << image_paths.size() << std::endl;
        return false;
    }
    test_images_.clear();
    for (const auto& path : image_paths) {
        cv::Mat image = cv::imread(path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            test_images_.clear();
            return false;
        }
        test_images_.push_back(image);
    }
    std::cout << "Loaded " << test_images_.size() << " test images." << std::endl;
    return true;
}

bool StitchingPipeline::LoadTestImagesFromMats(const std::vector<cv::Mat>& images) {
    if (images.size() != 3) {
        std::cerr << "Expected 3 images for stitching. Got " << images.size() << std::endl;
        return false;
    }

    test_images_.clear();
    for (const auto& img : images) {
        if (img.empty()) {
            std::cerr << "One of the provided RTSP images is empty." << std::endl;
            test_images_.clear();
            return false;
        }
        test_images_.push_back(img.clone());  // Store a copy
    }

    std::cout << "RTSP images loaded into stitching pipeline." << std::endl;
    return true;
}


cv::Mat StitchingPipeline::CreatePanoramaFromPrecomputed() {
    if (cameras_.size() != 3 || test_images_.size() != 3 || !loaded_ab_transform_.valid || !loaded_bc_transform_.valid) {
        std::cerr << "ERROR: Prerequisites for stitching not met. Load all data first." << std::endl;
        return cv::Mat();
    }
    
    std::cout << "=== STARTING PANORAMA CREATION ===" << std::endl;

    // 1. Rectify all three images
    rectified_images_.clear();
    for (size_t i = 0; i < test_images_.size(); ++i) {
        rectified_images_.push_back(RectifyImageFisheye(test_images_[i], cameras_[i].name, cameras_[i].intrinsics));
    }
    cv::Mat& img_izq = rectified_images_[0];
    cv::Mat& img_central = rectified_images_[1];
    cv::Mat& img_der = rectified_images_[2];
    std::cout << "1. All images rectified." << std::endl;

    // 2. Prepare transformations (Python script logic)
    // The AB transform is central -> izquierda. We need its inverse for izq -> central.
    cv::Mat transform_central_to_izq_3x3 = cv::Mat::eye(3, 3, CV_64F);
    loaded_ab_transform_.similarity.copyTo(transform_central_to_izq_3x3(cv::Rect(0, 0, 3, 2)));
    cv::Mat transform_izq_to_central_3x3;
    cv::invert(transform_central_to_izq_3x3, transform_izq_to_central_3x3);

    // The BC transform is derecha -> central, which is what we need.
    cv::Mat transform_der_to_central_3x3 = cv::Mat::eye(3, 3, CV_64F);
    loaded_bc_transform_.similarity.copyTo(transform_der_to_central_3x3(cv::Rect(0, 0, 3, 2)));
    std::cout << "2. Transformations prepared (AB inverted)." << std::endl;
    
    // 3. Calculate the global canvas size
    std::vector<cv::Point2f> corners_izq = { {0,0}, {(float)img_izq.cols, 0}, {(float)img_izq.cols, (float)img_izq.rows}, {0, (float)img_izq.rows} };
    std::vector<cv::Point2f> corners_central = { {0,0}, {(float)img_central.cols, 0}, {(float)img_central.cols, (float)img_central.rows}, {0, (float)img_central.rows} };
    std::vector<cv::Point2f> corners_der = { {0,0}, {(float)img_der.cols, 0}, {(float)img_der.cols, (float)img_der.rows}, {0, (float)img_der.rows} };

    std::vector<cv::Point2f> transformed_corners_izq;
    std::vector<cv::Point2f> transformed_corners_der;
    cv::perspectiveTransform(corners_izq, transformed_corners_izq, transform_izq_to_central_3x3);
    cv::perspectiveTransform(corners_der, transformed_corners_der, transform_der_to_central_3x3);

    std::vector<cv::Point2f> all_corners;
    all_corners.insert(all_corners.end(), corners_central.begin(), corners_central.end());
    all_corners.insert(all_corners.end(), transformed_corners_izq.begin(), transformed_corners_izq.end());
    all_corners.insert(all_corners.end(), transformed_corners_der.begin(), transformed_corners_der.end());

    cv::Rect bounding_box = cv::boundingRect(all_corners);
    std::cout << "3. Global canvas calculated: " << bounding_box.width << "x" << bounding_box.height << std::endl;

    // 4. Warp all images onto the final canvas
    cv::Mat offset_transform = cv::Mat::eye(3, 3, CV_64F);
    offset_transform.at<double>(0, 2) = -bounding_box.x;
    offset_transform.at<double>(1, 2) = -bounding_box.y;

    cv::Mat canvas(bounding_box.height, bounding_box.width, img_central.type(), cv::Scalar(0,0,0));

    // Warp central (our reference)
    cv::Mat warped_central;
    cv::warpPerspective(img_central, warped_central, offset_transform, bounding_box.size());
    
    // Warp izquierda
    cv::Mat warped_izq;
    cv::warpPerspective(img_izq, warped_izq, offset_transform * transform_izq_to_central_3x3, bounding_box.size());

    // Warp derecha
    cv::Mat warped_der;
    cv::warpPerspective(img_der, warped_der, offset_transform * transform_der_to_central_3x3, bounding_box.size());
    std::cout << "4. All images warped to final canvas." << std::endl;

    // 5. Blend the images together using maximum intensity (replicating python script)
    cv::Mat panorama;
    cv::max(warped_izq, warped_central, panorama);
    cv::max(panorama, warped_der, panorama);
    std::cout << "5. Images blended using cv::max." << std::endl;

    // 6. Crop final image to remove black borders
    cv::Mat gray;
    cv::cvtColor(panorama, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, 1, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        double max_area = 0;
        size_t max_area_idx = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_area_idx = i;
            }
        }
        cv::Rect crop_rect = cv::boundingRect(contours[max_area_idx]);
        std::cout << "6. Cropping panorama to content." << std::endl;
        return panorama(crop_rect);
    }
    
    std::cout << "6. No content found to crop, returning full canvas." << std::endl;
    return panorama;
}

cv::Mat StitchingPipeline::CreatePanoramaWithCustomTransforms(const cv::Mat& custom_ab_transform, const cv::Mat& custom_bc_transform) {
    if (cameras_.size() != 3 || test_images_.size() != 3) {
        std::cerr << "ERROR: Prerequisites for stitching not met. Load all data first." << std::endl;
        return cv::Mat();
    }
    
    // std::cout << "=== STARTING PANORAMA CREATION WITH CUSTOM TRANSFORMS ===" << std::endl;

    // 1. Rectify all three images (reuse existing rectified images if available)
    if (rectified_images_.size() != 3) {
        rectified_images_.clear();
        for (size_t i = 0; i < test_images_.size(); ++i) {
            rectified_images_.push_back(RectifyImageFisheye(test_images_[i], cameras_[i].name, cameras_[i].intrinsics));
        }
    }
    cv::Mat& img_izq = rectified_images_[0];
    cv::Mat& img_central = rectified_images_[1];
    cv::Mat& img_der = rectified_images_[2];
    // std::cout << "1. Using rectified images." << std::endl;

    // 2. Use custom transformations
    // Apply custom AB transform to the loaded transform
    cv::Mat base_ab_transform_3x3 = cv::Mat::eye(3, 3, CV_64F);
    if (loaded_ab_transform_.valid) {
        loaded_ab_transform_.similarity.copyTo(base_ab_transform_3x3(cv::Rect(0, 0, 3, 2)));
    }
    cv::Mat combined_ab_transform = custom_ab_transform * base_ab_transform_3x3;
    
    cv::Mat transform_central_to_izq_3x3 = combined_ab_transform;
    cv::Mat transform_izq_to_central_3x3;
    cv::invert(transform_central_to_izq_3x3, transform_izq_to_central_3x3);

    // Apply custom BC transform to the loaded transform  
    cv::Mat base_bc_transform_3x3 = cv::Mat::eye(3, 3, CV_64F);
    if (loaded_bc_transform_.valid) {
        loaded_bc_transform_.similarity.copyTo(base_bc_transform_3x3(cv::Rect(0, 0, 3, 2)));
    }
    cv::Mat transform_der_to_central_3x3 = custom_bc_transform * base_bc_transform_3x3;
    // std::cout << "2. Custom transformations applied." << std::endl;
    
    // 3. Calculate the global canvas size
    std::vector<cv::Point2f> corners_izq = { {0,0}, {(float)img_izq.cols, 0}, {(float)img_izq.cols, (float)img_izq.rows}, {0, (float)img_izq.rows} };
    std::vector<cv::Point2f> corners_central = { {0,0}, {(float)img_central.cols, 0}, {(float)img_central.cols, (float)img_central.rows}, {0, (float)img_central.rows} };
    std::vector<cv::Point2f> corners_der = { {0,0}, {(float)img_der.cols, 0}, {(float)img_der.cols, (float)img_der.rows}, {0, (float)img_der.rows} };

    std::vector<cv::Point2f> transformed_corners_izq;
    std::vector<cv::Point2f> transformed_corners_der;
    cv::perspectiveTransform(corners_izq, transformed_corners_izq, transform_izq_to_central_3x3);
    cv::perspectiveTransform(corners_der, transformed_corners_der, transform_der_to_central_3x3);

    std::vector<cv::Point2f> all_corners;
    all_corners.insert(all_corners.end(), corners_central.begin(), corners_central.end());
    all_corners.insert(all_corners.end(), transformed_corners_izq.begin(), transformed_corners_izq.end());
    all_corners.insert(all_corners.end(), transformed_corners_der.begin(), transformed_corners_der.end());

    cv::Rect bounding_box = cv::boundingRect(all_corners);
    // std::cout << "3. Global canvas calculated: " << bounding_box.width << "x" << bounding_box.height << std::endl;

    // 4. Warp all images onto the final canvas
    cv::Mat offset_transform = cv::Mat::eye(3, 3, CV_64F);
    offset_transform.at<double>(0, 2) = -bounding_box.x;
    offset_transform.at<double>(1, 2) = -bounding_box.y;

    cv::Mat canvas(bounding_box.height, bounding_box.width, img_central.type(), cv::Scalar(0,0,0));

    // Warp central (our reference)
    cv::Mat warped_central;
    cv::warpPerspective(img_central, warped_central, offset_transform, bounding_box.size());
    
    // Warp izquierda
    cv::Mat warped_izq;
    cv::warpPerspective(img_izq, warped_izq, offset_transform * transform_izq_to_central_3x3, bounding_box.size());

    // Warp derecha
    cv::Mat warped_der;
    cv::warpPerspective(img_der, warped_der, offset_transform * transform_der_to_central_3x3, bounding_box.size());
    // std::cout << "4. All images warped to final canvas." << std::endl;

    // 5. Blend the images together using the selected blending mode
    cv::Mat panorama;
    if (blending_mode_ == BlendingMode::AVERAGE) {
        cv::max(warped_izq, warped_central, panorama);
        cv::max(panorama, warped_der, panorama);
        // std::cout << "5. Images blended using cv::max." << std::endl;
    } else {
        // Use feathering blending
        panorama = warped_central.clone();
        BlendInPlace(panorama, warped_izq, BlendingMode::FEATHERING);
        BlendInPlace(panorama, warped_der, BlendingMode::FEATHERING);
        // std::cout << "5. Images blended using feathering." << std::endl;
    }

    // 6. Crop final image to remove black borders
    cv::Mat gray;
    cv::cvtColor(panorama, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, 1, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        double max_area = 0;
        size_t max_area_idx = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_area_idx = i;
            }
        }
        cv::Rect crop_rect = cv::boundingRect(contours[max_area_idx]);
        // std::cout << "6. Cropping panorama to content." << std::endl;
        return panorama(crop_rect);
    }
    
    std::cout << "6. No content found to crop, returning full canvas." << std::endl;
    return panorama;
}

void StitchingPipeline::SetBlendingMode(BlendingMode mode) {
    blending_mode_ = mode;
}

// --- PRIVATE HELPER METHODS ---

void StitchingPipeline::PrecomputeRectificationMaps() {
    map1_.clear();
    map2_.clear();

    for (const auto& cam : cameras_) {
        cv::Mat map1, map2;
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);  // No rectification rotation
        cv::Size size(cam.intrinsics.image_width, cam.intrinsics.image_height);

        cv::Mat new_K;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        cam.intrinsics.camera_matrix,
        cam.intrinsics.distortion_coeffs,
        size,
        R,
        new_K,
        0.4  // balance, adjust if needed
    );

        cv::fisheye::initUndistortRectifyMap(
            cam.intrinsics.camera_matrix,
            cam.intrinsics.distortion_coeffs,
            R,
            new_K,  // You can use new_K for zooming out/in
            size,
            CV_16SC2,
            map1,
            map2
        );

        map1_[cam.name] = map1;
        map2_[cam.name] = map2;
    }

    maps_ready_ = true;
}


void StitchingPipeline::BlendInPlace(cv::Mat& base, const cv::Mat& overlay, BlendingMode mode) {
    if (base.empty() || overlay.empty()) return;

    // Create masks for non-black areas
    cv::Mat base_mask, overlay_mask;
    cv::cvtColor(base, base_mask, cv::COLOR_BGR2GRAY);
    cv::cvtColor(overlay, overlay_mask, cv::COLOR_BGR2GRAY);
    cv::threshold(base_mask, base_mask, 1, 255, cv::THRESH_BINARY);
    cv::threshold(overlay_mask, overlay_mask, 1, 255, cv::THRESH_BINARY);

    // Find the intersection (overlap region)
    cv::Mat intersection = base_mask & overlay_mask;
    if (cv::countNonZero(intersection) == 0) { // No overlap
        overlay.copyTo(base, overlay_mask); // Just copy non-overlapping parts
        return;
    }

    if (mode == BlendingMode::AVERAGE) {
        // Simple average blending in the intersection
        cv::Mat blended_roi;
        cv::addWeighted(base, 0.5, overlay, 0.5, 0.0, blended_roi);
        blended_roi.copyTo(base, intersection);
        // Also copy the parts of the overlay that don't overlap with the base
        overlay.copyTo(base, overlay_mask & ~base_mask);

    } else if (mode == BlendingMode::FEATHERING) {
        // Feathering based on distance transform
        cv::Rect roi = cv::boundingRect(intersection);
        if (roi.width <= 0 || roi.height <= 0) return;

        // We need two distance transforms, one for each image's non-overlapping edge within the intersection
        cv::Mat dist_base, dist_overlay;
        cv::distanceTransform(intersection, dist_base, cv::DIST_L2, 3);
        cv::distanceTransform(overlay_mask, dist_overlay, cv::DIST_L2, 3);

        cv::Mat weights_overlay(dist_base.size(), CV_32F);
        for(int r = 0; r < weights_overlay.rows; ++r) {
            for (int c = 0; c < weights_overlay.cols; ++c) {
                float d_base = dist_base.at<float>(r,c);
                float d_overlay = dist_overlay.at<float>(r,c);
                float total = d_base + d_overlay;
                weights_overlay.at<float>(r,c) = (total > 0) ? (d_base / total) : 0.0f;
            }
        }

        // Apply feathering only within the intersection ROI
        cv::Mat weights_overlay_roi = weights_overlay(roi);
        cv::Mat base_roi = base(roi);
        cv::Mat overlay_roi = overlay(roi);
        cv::Mat intersection_roi = intersection(roi);

        for (int r = 0; r < base_roi.rows; ++r) {
            for (int c = 0; c < base_roi.cols; ++c) {
                if (intersection_roi.at<uchar>(r, c) > 0) { // Only blend in the intersection
                    float w = weights_overlay_roi.at<float>(r, c);
                    base_roi.at<cv::Vec3b>(r, c) = 
                        base_roi.at<cv::Vec3b>(r, c) * (1.0 - w) + 
                        overlay_roi.at<cv::Vec3b>(r, c) * w;
                }
            }
        }
        // Copy the non-overlapping parts of the overlay to the base
        overlay.copyTo(base, overlay_mask & ~base_mask);
    }
}

cv::Mat StitchingPipeline::RectifyImageFisheye(const cv::Mat& image, const std::string& name, const CameraIntrinsics& intrinsics) {
    if (!maps_ready_ || map1_.count(name) == 0) {
        std::cerr << "Rectification maps not ready for: " << intrinsics.model << std::endl;
        return image.clone();  // Return unrectified image
    }

    cv::Mat rectified;
    cv::remap(image, rectified, map1_[name], map2_[name], cv::INTER_LINEAR);
    return rectified;
}

const CameraIntrinsics& StitchingPipeline::GetCameraIntrinsicsByName(const std::string& name) const {
    for (const auto& cam : cameras_) {
        if (cam.name == name) {
            return cam.intrinsics;
        }
    }
    throw std::runtime_error("Camera intrinsics not found for: " + name);
}
