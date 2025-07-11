// rtsp_stitcher.cpp
#include "stitching.hh"
#include <opencv2/opencv.hpp>
#include <csignal>
#include <atomic>
#include <thread>
#include <filesystem>
#include <iostream>
#include <mutex>

using namespace std;
namespace fs = std::filesystem;

atomic<bool> stop_requested(false);

void signal_handler(int signal) {
    if (signal == SIGINT) {
        stop_requested = true;
        cout << "\n[INFO] Ctrl+C received. Finishing recording before exit...\n";
    }
}

// Captures frames from an RTSP stream and stores them in a shared buffer
void CaptureAndSave(const string& url, const string& output_path, vector<cv::Mat>& buffer, mutex& buffer_mutex, int cam_id) {
    cv::VideoCapture cap(url);
    if (!cap.isOpened()) {
        cerr << "[CAM " << cam_id << "] Failed to open: " << url << endl;
        return;
    }

    cout << "[CAM " << cam_id << "] Started recording." << endl;
    while (!stop_requested) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;
        {
            lock_guard<mutex> lock(buffer_mutex);
            buffer.push_back(frame.clone());
        }
    }
    cap.release();
    cout << "[CAM " << cam_id << "] Stopped recording." << endl;
}

// Writes frames to a video file
void SaveVideo(const vector<cv::Mat>& frames, const string& path, double fps) {
    if (frames.empty()) return;
    int width = frames[0].cols;
    int height = frames[0].rows;
    cv::VideoWriter writer(path, cv::VideoWriter::fourcc('X','V','I','D'), fps, cv::Size(width, height));
    for (const auto& frame : frames) writer.write(frame);
    writer.release();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <rtsp_left> <rtsp_center> <rtsp_right>" << endl;
        return -1;
    }

    signal(SIGINT, signal_handler);
    fs::create_directory("output");
    fs::create_directory("debug");

    vector<string> rtsp_links = { argv[1], argv[2], argv[3] };
    vector<string> raw_paths = {
        "output/raw_left.avi",
        "output/raw_center.avi",
        "output/raw_right.avi"
    };
    vector<string> rectified_paths = {
        "output/rectified_left.avi",
        "output/rectified_center.avi",
        "output/rectified_right.avi"
    };

    vector<vector<cv::Mat>> buffers(3);
    vector<mutex> buffer_mutexes(3);
    vector<thread> threads;

    // Launch threads to capture all three RTSP streams
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(CaptureAndSave, rtsp_links[i], raw_paths[i], ref(buffers[i]), ref(buffer_mutexes[i]), i);
    }
    for (auto& t : threads) t.join();

    // Load calibration data
    double fps = 15.0;
    StitchingPipeline pipeline;
    if (!pipeline.LoadIntrinsicsData("intrinsic.json") ||
        !pipeline.LoadExtrinsicsData("extrinsic.json")) {
        cerr << "[ERROR] Failed to load calibration data." << endl;
        return -1;
    }

    // Setup video writer for panorama
    cv::VideoWriter pano_writer;
    bool pano_writer_ready = false;
    size_t frame_count = min({ buffers[0].size(), buffers[1].size(), buffers[2].size() });

    // Create vectors to store rectified frames for saving later
    vector<vector<cv::Mat>> rectified_buffers(3);

    for (size_t f = 0; f < frame_count; ++f) {
        // Instead of manually rectifying, let the pipeline handle it
        // Just pass the raw frames directly to LoadTestImagesFromMats
        vector<cv::Mat> raw_frames = { buffers[0][f], buffers[1][f], buffers[2][f] };
        
        pipeline.SetBlendingMode(BlendingMode::AVERAGE);

        if (!pipeline.LoadTestImagesFromMats(raw_frames)) {
            cerr << "[ERROR] Failed to load raw images into pipeline." << endl;
            continue;
        }

        // Use the same custom transformation matrices as in GUI
        // AB Transform (izquierda -> central)
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
        
        // BC Transform (derecha -> central)
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

        // Generate panorama
        cv::Mat pano = pipeline.CreatePanoramaWithCustomTransforms(ab_custom, bc_custom);
        if (pano.empty()) {
            cerr << "[WARNING] Panorama was empty at frame " << f << endl;
            continue;
        }

        // Initialize writer with panorama size
        if (!pano_writer_ready) {
            pano_writer.open("output/panorama_result.mp4", cv::VideoWriter::fourcc('a','v','c','1'), fps, pano.size());
            if (!pano_writer.isOpened()) {
                cerr << "[ERROR] Could not open output file for panorama." << endl;
                return -1;
            }
            pano_writer_ready = true;
        }

        pano_writer.write(pano);
        
        // Save debug frame every 30 frames to avoid too many files
        if (f % 30 == 0) {
            cv::imwrite("debug/pano_" + to_string(f) + ".png", pano);
        }

        // If you need rectified frames for saving, get them from the pipeline
        // You'll need to add a method to your StitchingPipeline class to get rectified images
        // or manually rectify here for saving purposes only
        if (f == 0) { // Only rectify for the first frame to save one example
            vector<cv::Mat> rectified(3);
            rectified[0] = pipeline.RectifyImageFisheye(buffers[0][f], "izquierda", pipeline.GetCameraIntrinsicsByName("izquierda"));
            rectified[1] = pipeline.RectifyImageFisheye(buffers[1][f], "central", pipeline.GetCameraIntrinsicsByName("central"));
            rectified[2] = pipeline.RectifyImageFisheye(buffers[2][f], "derecha", pipeline.GetCameraIntrinsicsByName("derecha"));
            
            for (int i = 0; i < 3; ++i) {
                rectified_buffers[i].push_back(rectified[i]);
            }
        }
    }

    // Save individual video streams
    for (int i = 0; i < 3; ++i) SaveVideo(buffers[i], raw_paths[i], fps);
    
    // Save rectified videos (only if we have rectified frames)
    for (int i = 0; i < 3; ++i) {
        if (!rectified_buffers[i].empty()) {
            SaveVideo(rectified_buffers[i], rectified_paths[i], fps);
        }
    }
    
    if (pano_writer_ready) pano_writer.release();

    cout << "\n[INFO] Finished. All videos saved in /output.\n" << endl;
    return 0;
}