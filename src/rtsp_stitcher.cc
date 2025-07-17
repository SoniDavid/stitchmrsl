// rtsp_stitcher.cpp
#include "stitching.hh"
#include <opencv2/opencv.hpp>
#include <csignal>
#include <atomic>
#include <thread>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <chrono>

using namespace std;
namespace fs = std::filesystem;

atomic<bool> stop_requested(false);

void signal_handler(int signal) {
    if (signal == SIGINT) {
        stop_requested = true;
        cout << "\n[INFO] Ctrl+C received. Finishing current recording before exit...\n";
    }
}

// Step 1: Capture RTSP streams and save as raw video files
void CaptureRTSPToFile(const string& url, const string& output_path, int cam_id) {
    cv::VideoCapture cap(url);
    if (!cap.isOpened()) {
        cerr << "[CAM " << cam_id << "] Failed to open: " << url << endl;
        return;
    }

    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 15.0; // Default fallback
    
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    cout << "[CAM " << cam_id << "] Properties: " << width << "x" << height << " @ " << fps << " FPS" << endl;
    
    // Create video writer
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        cerr << "[CAM " << cam_id << "] Failed to create video writer for: " << output_path << endl;
        return;
    }

    cout << "[CAM " << cam_id << "] Started recording to: " << output_path << endl;
    
    int frame_count = 0;
    auto start_time = chrono::steady_clock::now();
    
    while (!stop_requested) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            cerr << "[CAM " << cam_id << "] Failed to read frame or stream ended" << endl;
            break;
        }
        
        writer.write(frame);
        frame_count++;
        
        // Progress update every 5 seconds
        if (frame_count % (static_cast<int>(fps) * 5) == 0) {
            auto elapsed = chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start_time).count();
            cout << "[CAM " << cam_id << "] Recorded " << frame_count << " frames (" << elapsed << "s)" << endl;
        }
    }
    
    writer.release();
    cap.release();
    cout << "[CAM " << cam_id << "] Stopped recording. Total frames: " << frame_count << endl;
}

// Step 2: Extract frames from video files
bool ExtractFramesFromVideo(const string& video_path, const string& frames_dir, const string& prefix) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Failed to open video: " << video_path << endl;
        return false;
    }
    
    fs::create_directories(frames_dir);
    
    int frame_count = 0;
    cv::Mat frame;
    
    cout << "[INFO] Extracting frames from: " << video_path << endl;
    
    while (cap.read(frame)) {
        if (frame.empty()) break;
        
        // Create frame filename with zero-padded numbers
        stringstream ss;
        ss << frames_dir << "/" << prefix << "_" << setfill('0') << setw(6) << frame_count << ".png";
        string frame_path = ss.str();
        
        if (!cv::imwrite(frame_path, frame)) {
            cerr << "[ERROR] Failed to save frame: " << frame_path << endl;
            cap.release();
            return false;
        }
        
        frame_count++;
        
        // Progress update
        if (frame_count % 150 == 0) {
            cout << "[INFO] Extracted " << frame_count << " frames from " << prefix << endl;
        }
    }
    
    cap.release();
    cout << "[INFO] Finished extracting " << frame_count << " frames from " << prefix << endl;
    return true;
}

// Step 3: Process frames and create panorama video
bool ProcessFramesToPanorama(const string& frames_base_dir, const string& output_video_path, 
                           StitchingPipeline& pipeline, double fps = 15.0) {
    
    // Find the minimum number of frames across all cameras
    vector<string> camera_names = {"left", "central", "right"};
    vector<string> frame_dirs = {
        frames_base_dir + "/left",
        frames_base_dir + "/central", 
        frames_base_dir + "/right"
    };
    
    int min_frames = INT_MAX;
    for (const auto& dir : frame_dirs) {
        int count = 0;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".png") count++;
        }
        min_frames = min(min_frames, count);
        cout << "[INFO] Found " << count << " frames in " << dir << endl;
    }
    
    if (min_frames == 0) {
        cerr << "[ERROR] No frames found in one or more directories" << endl;
        return false;
    }
    
    cout << "[INFO] Processing " << min_frames << " synchronized frames" << endl;
    
    // Setup custom transformation matrices (same as original)
    cv::Mat ab_custom = cv::Mat::eye(3, 3, CV_64F);
    float ab_rad = 0.800f * CV_PI / 180.0f;
    float ab_cos = cos(ab_rad);
    float ab_sin = sin(ab_rad);
    float ab_scale_x = 0.9877f;
    float ab_scale_y = 1.0f;
    float ab_translation_x = -2.8f;
    float ab_translation_y = 0.0f;

    ab_custom.at<double>(0, 0) = ab_cos * ab_scale_x; 
    ab_custom.at<double>(0, 1) = -ab_sin * ab_scale_x;
    ab_custom.at<double>(0, 2) = ab_translation_x; 
    ab_custom.at<double>(1, 0) = ab_sin * ab_scale_y; 
    ab_custom.at<double>(1, 1) = ab_cos * ab_scale_y;
    ab_custom.at<double>(1, 2) = ab_translation_y; 
    
    cv::Mat bc_custom = cv::Mat::eye(3, 3, CV_64F);
    float bc_rad = 0.800f * CV_PI / 180.0f;
    float bc_cos = cos(bc_rad);
    float bc_sin = sin(bc_rad);
    float bc_scale_x = 1.0f;
    float bc_scale_y = 1.0f;
    float bc_translation_x = 21.1f;
    float bc_translation_y = 0.0f;
    
    bc_custom.at<double>(0, 0) = bc_cos * bc_scale_x;
    bc_custom.at<double>(0, 1) = -bc_sin * bc_scale_x;
    bc_custom.at<double>(0, 2) = bc_translation_x; 
    bc_custom.at<double>(1, 0) = bc_sin * bc_scale_y;
    bc_custom.at<double>(1, 1) = bc_cos * bc_scale_y;
    bc_custom.at<double>(1, 2) = bc_translation_y; 

    pipeline.SetBlendingMode(BlendingMode::AVERAGE);
    
    cv::VideoWriter pano_writer;
    bool writer_initialized = false;
    
    for (int frame_idx = 0; frame_idx < min_frames; frame_idx++) {
        cout << "[INFO] Processing frame " << frame_idx << "/" << min_frames << endl;
        
        // CRITICAL: Clear pipeline cache before each frame
        pipeline.ClearCache();
        
        // Load synchronized frames
        vector<cv::Mat> frames(3);
        vector<string> frame_paths(3);
        bool frames_loaded = true;
        
        for (int cam = 0; cam < 3; cam++) {
            stringstream ss;
            ss << frame_dirs[cam] << "/" << camera_names[cam] << "_" << setfill('0') << setw(6) << frame_idx << ".png";
            frame_paths[cam] = ss.str();
            
            // Check if file exists first
            if (!fs::exists(frame_paths[cam])) {
                cerr << "[ERROR] Frame file does not exist: " << frame_paths[cam] << endl;
                frames_loaded = false;
                break;
            }
            
            frames[cam] = cv::imread(frame_paths[cam], cv::IMREAD_COLOR);
            if (frames[cam].empty()) {
                cerr << "[ERROR] Failed to load frame: " << frame_paths[cam] << endl;
                frames_loaded = false;
                break;
            }
            
            // Verify frame dimensions
            if (frames[cam].rows == 0 || frames[cam].cols == 0) {
                cerr << "[ERROR] Invalid frame dimensions for: " << frame_paths[cam] << endl;
                frames_loaded = false;
                break;
            }
        }
        
        if (!frames_loaded) {
            cerr << "[ERROR] Skipping frame " << frame_idx << " due to loading issues" << endl;
            continue;
        }
        
        // Debug: Save input frames for first few frames
        if (frame_idx < 5) {
            fs::create_directories("debug/input");
            for (int cam = 0; cam < 3; cam++) {
                string debug_path = "debug/input/" + camera_names[cam] + "_frame_" + to_string(frame_idx) + ".png";
                cv::imwrite(debug_path, frames[cam]);
            }
            cout << "[DEBUG] Saved input frames for frame " << frame_idx << endl;
        }
        
        // Load fresh frames into pipeline
        if (!pipeline.LoadTestImagesFromMats(frames)) {
            cerr << "[ERROR] Failed to load frames into pipeline for frame " << frame_idx << endl;
            continue;
        }
        
        // Generate panorama with fresh data
        cv::Mat pano = pipeline.CreatePanoramaWithCustomTransforms(ab_custom, bc_custom);
        if (pano.empty()) {
            cerr << "[WARNING] Empty panorama at frame " << frame_idx << endl;
            continue;
        }
        
        // Debug: Check if panorama is actually different
        static cv::Mat prev_pano;
        if (!prev_pano.empty() && frame_idx > 0) {
            cv::Mat diff;
            cv::absdiff(pano, prev_pano, diff);
            cv::Scalar diff_sum = cv::sum(diff);
            double total_diff = diff_sum[0] + diff_sum[1] + diff_sum[2];
            
            if (total_diff < 1000) { // Very small difference threshold
                cerr << "[WARNING] Frame " << frame_idx << " is very similar to previous frame (diff: " << total_diff << ")" << endl;
            } else {
                cout << "[DEBUG] Frame " << frame_idx << " difference from previous: " << total_diff << endl;
            }
        }
        prev_pano = pano.clone();
        
        // Initialize video writer on first successful frame
        if (!writer_initialized) {
            pano_writer.open(output_video_path, cv::VideoWriter::fourcc('M','J','P','G'), fps, pano.size());
            if (!pano_writer.isOpened()) {
                cerr << "[ERROR] Could not open output video file: " << output_video_path << endl;
                return false;
            }
            writer_initialized = true;
            cout << "[INFO] Initialized video writer: " << pano.size() << " @ " << fps << " FPS" << endl;
        }
        
        // Write frame to video
        pano_writer.write(pano);
        
        // Save debug frame more frequently for debugging
        if (frame_idx % 30 == 0) {
            fs::create_directories("debug/output");
            cv::imwrite("debug/output/pano_" + to_string(frame_idx) + ".png", pano);
        }
        
        // Clear frames to free memory
        frames.clear();
        
        // Progress update
        if (frame_idx % 30 == 0) {
            cout << "[INFO] Processed frame " << frame_idx << "/" << min_frames << endl;
        }
    }
    
    if (writer_initialized) {
        pano_writer.release();
        cout << "[INFO] Panorama video saved: " << output_video_path << endl;
    }
    
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <rtsp_left> <rtsp_center> <rtsp_right>" << endl;
        return -1;
    }

    signal(SIGINT, signal_handler);
    
    // Create directory structure
    fs::create_directories("output/raw");
    fs::create_directories("output/frames/left");
    fs::create_directories("output/frames/central");
    fs::create_directories("output/frames/right");
    fs::create_directories("debug");

    vector<string> rtsp_links = { argv[1], argv[2], argv[3] };
    vector<string> raw_paths = {
        "output/raw/left.mp4",
        "output/raw/central.mp4",
        "output/raw/right.mp4"
    };
    
    cout << "\n=== STEP 1: CAPTURING RTSP STREAMS ===" << endl;
    
    // Step 1: Capture RTSP streams in parallel
    vector<thread> capture_threads;
    for (int i = 0; i < 3; ++i) {
        capture_threads.emplace_back(CaptureRTSPToFile, rtsp_links[i], raw_paths[i], i);
    }
    
    // Wait for all capture threads
    for (auto& t : capture_threads) {
        t.join();
    }
    
    if (stop_requested) {
        cout << "[INFO] Capture interrupted by user" << endl;
    }
    
    cout << "\n=== STEP 2: EXTRACTING FRAMES ===" << endl;
    
    // Step 2: Extract frames from videos
    vector<string> camera_names = {"left", "central", "right"};
    vector<string> frame_dirs = {
        "output/frames/left",
        "output/frames/central",
        "output/frames/right"
    };
    
    for (int i = 0; i < 3; ++i) {
        if (!fs::exists(raw_paths[i])) {
            cerr << "[ERROR] Raw video not found: " << raw_paths[i] << endl;
            continue;
        }
        
        if (!ExtractFramesFromVideo(raw_paths[i], frame_dirs[i], camera_names[i])) {
            cerr << "[ERROR] Failed to extract frames from: " << raw_paths[i] << endl;
            return -1;
        }
    }
    
    cout << "\n=== STEP 3: PROCESSING PANORAMA ===" << endl;
    
    // Step 3: Load calibration and process frames
    StitchingPipeline pipeline;
    if (!pipeline.LoadIntrinsicsData("intrinsic.json") ||
        !pipeline.LoadExtrinsicsData("extrinsic.json")) {
        cerr << "[ERROR] Failed to load calibration data." << endl;
        return -1;
    }
    
    double fps = 15.0; // You might want to read this from the original videos
    if (!ProcessFramesToPanorama("output/frames", "output/panorama_result.mp4", pipeline, fps)) {
        cerr << "[ERROR] Failed to create panorama video" << endl;
        return -1;
    }
    
    cout << "\n=== PROCESSING COMPLETE ===" << endl;
    cout << "Raw videos: output/raw/ (MP4 format)" << endl;
    cout << "Extracted frames: output/frames/" << endl;
    cout << "Final panorama: output/panorama_result.mp4" << endl;
    cout << "Debug frames: debug/" << endl;
    
    return 0;
}