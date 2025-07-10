// rtsp_stitcher.cpp
#include "stitching.hh"
#include <opencv2/opencv.hpp>
#include <csignal>
#include <atomic>
#include <thread>
#include <filesystem>
#include <iostream>

using namespace std;
namespace fs = std::filesystem;

atomic<bool> stop_requested(false);

void signal_handler(int signal) {
    if (signal == SIGINT) {
        stop_requested = true;
        cout << "\n[INFO] Ctrl+C received. Finishing recording before exit...\n";
    }
}

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

    int pano_frames_written = 0;

    signal(SIGINT, signal_handler);
    fs::create_directory("output");

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

    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(CaptureAndSave, rtsp_links[i], raw_paths[i], ref(buffers[i]), ref(buffer_mutexes[i]), i);
    }
    for (auto& t : threads) t.join();

    double fps = 15.0;
    StitchingPipeline pipeline;
    if (!pipeline.LoadIntrinsicsData("intrinsic.json") ||
        !pipeline.LoadExtrinsicsData("extrinsic.json")) {
        cerr << "[ERROR] Failed to load calibration data." << endl;
        return -1;
    }

    cv::VideoWriter pano_writer;
    bool pano_writer_ready = false;

    size_t frame_count = min({ buffers[0].size(), buffers[1].size(), buffers[2].size() });

    for (size_t f = 0; f < frame_count; ++f) {
        vector<cv::Mat> rectified(3);
        rectified[0] = pipeline.RectifyImageFisheye(buffers[0][f], "izquierda", pipeline.GetCameraIntrinsicsByName("izquierda"));
        rectified[1] = pipeline.RectifyImageFisheye(buffers[1][f], "central", pipeline.GetCameraIntrinsicsByName("central"));
        rectified[2] = pipeline.RectifyImageFisheye(buffers[2][f], "derecha", pipeline.GetCameraIntrinsicsByName("derecha"));

        for (int i = 0; i < 3; ++i) buffers[i][f] = rectified[i];

        if (pipeline.LoadTestImagesFromMats(rectified)) {
            cv::Mat ab_custom = cv::Mat::eye(3, 3, CV_64F);
            float ab_rad = 0.800f * CV_PI / 180.0f;
            ab_custom.at<double>(0, 0) = cos(ab_rad) * 0.9877;
            ab_custom.at<double>(0, 1) = -sin(ab_rad) * 0.9877;
            ab_custom.at<double>(0, 2) = -2.8;
            ab_custom.at<double>(1, 0) = sin(ab_rad);
            ab_custom.at<double>(1, 1) = cos(ab_rad);

            cv::Mat bc_custom = cv::Mat::eye(3, 3, CV_64F);
            float bc_rad = 0.0f;
            bc_custom.at<double>(0, 0) = cos(bc_rad);
            bc_custom.at<double>(0, 1) = -sin(bc_rad);
            bc_custom.at<double>(0, 2) = 21.1;
            bc_custom.at<double>(1, 0) = sin(bc_rad);
            bc_custom.at<double>(1, 1) = cos(bc_rad);

            pipeline.SetBlendingMode(BlendingMode::AVERAGE);
            cv::Mat pano = pipeline.CreatePanoramaWithCustomTransforms(ab_custom, bc_custom);

            if (!pano.empty()) {
                if (!pano_writer_ready) {
                    const auto pano_size = pano.size();
                    string pano_path = "output/panorama_result.mp4";

                    pano_writer.open(pano_path, cv::VideoWriter::fourcc('a','v','c','1'), fps, pano_size);
                    if (!pano_writer.isOpened()) {
                        cerr << "[WARNING] Could not open .mp4 with avc1 codec. Falling back to .avi (XVID).\n";
                        pano_path = "output/panorama_result.avi";
                        pano_writer.open(pano_path, cv::VideoWriter::fourcc('X','V','I','D'), fps, pano_size);
                    }

                    if (!pano_writer.isOpened()) {
                        cerr << "[ERROR] Failed to open panorama writer after fallback.\n";
                        return -1;
                    }

                    pano_writer_ready = true;
                    cout << "[INFO] Panorama writer initialized with size: " << pano_size << endl;
                }

                pano_writer.write(pano);
            } else {
                cerr << "[WARN] Empty pano frame skipped.\n";
            }
        }
        pano_frames_written++;
    }

    cout << "[INFO] Total panorama frames written: " << pano_frames_written << endl;

    for (int i = 0; i < 3; ++i) SaveVideo(buffers[i], raw_paths[i], fps);
    for (int i = 0; i < 3; ++i) SaveVideo(buffers[i], rectified_paths[i], fps);
    if (pano_writer_ready) pano_writer.release();

    cout << "\n[INFO] Finished. All videos saved in /output." << endl;
    return 0;
}
