#include "stitching.hh"
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <filesystem>
#include <fstream>

using namespace std;
namespace fs = std::filesystem;

// Global flag
atomic<bool> running(true);

void CaptureStream(const string& url, const string& out_path, int index) {
    cv::VideoCapture cap(url);
    if (!cap.isOpened()) {
        cerr << "Failed to open RTSP stream: " << url << endl;
        running = false;
        return;
    }

    int fourcc = cv::VideoWriter::fourcc('X','V','I','D');
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps < 1.0) fps = 15.0;
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter writer(out_path, fourcc, fps, cv::Size(width, height));

    if (!writer.isOpened()) {
        cerr << "Failed to open video writer: " << out_path << endl;
        running = false;
        return;
    }

    cout << "[CAM " << index << "] Recording started..." << endl;

    while (running) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;
        writer.write(frame);
        cv::imshow("CAM " + to_string(index), frame);
        if (cv::waitKey(10) == 27) running = false;
    }

    cap.release();
    writer.release();
    cout << "[CAM " << index << "] Recording finished." << endl;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <rtsp_left> <rtsp_center> <rtsp_right>" << endl;
        return -1;
    }

    fs::create_directory("output");

    vector<string> rtsp_links = { argv[1], argv[2], argv[3] };
    vector<string> video_paths = {
        "output/raw_left.avi",
        "output/raw_center.avi",
        "output/raw_right.avi"
    };

    vector<thread> threads;
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(CaptureStream, rtsp_links[i], video_paths[i], i);
    }
    for (auto& t : threads) t.join();

    StitchingPipeline pipeline;
    if (!pipeline.LoadIntrinsicsData("calibration/intrinsics.json") ||
        !pipeline.LoadExtrinsicsData("calibration/extrinsics.json")) {
        cerr << "Calibration loading failed!" << endl;
        return -1;
    }

    vector<string> rectified_paths = {
        "output/rectified_left.avi",
        "output/rectified_center.avi",
        "output/rectified_right.avi"
    };

    cv::VideoWriter panorama_writer;
    bool panorama_writer_initialized = false;

    for (int i = 0; i < 3; ++i) {
        cv::VideoCapture cap(video_paths[i]);
        if (!cap.isOpened()) {
            cerr << "Failed to reopen " << video_paths[i] << endl;
            return -1;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        cv::VideoWriter writer(rectified_paths[i],
            cv::VideoWriter::fourcc('X','V','I','D'),
            fps, cv::Size(width, height));

        if (!writer.isOpened()) {
            cerr << "Failed to create writer for " << rectified_paths[i] << endl;
            return -1;
        }

        vector<cv::Mat> frames(3);
        ifstream testcap(video_paths[0]);
        while (cap.read(frames[i])) {
            string name = (i == 0) ? "izquierda" : (i == 1) ? "central" : "derecha";
            auto intr = pipeline.GetCameraIntrinsicsByName(name);
            auto rectified = pipeline.RectifyImageFisheye(frames[i], name, intr);
            writer.write(rectified);
            frames[i] = rectified;

            if (i == 2) {
                if (!pipeline.LoadTestImagesFromMats(frames)) continue;

                cv::Mat ab_custom = cv::Mat::eye(3, 3, CV_64F);
                float ab_rad = 0.800f * CV_PI / 180.0f;
                ab_custom.at<double>(0, 0) = cos(ab_rad) * 0.9877;
                ab_custom.at<double>(0, 1) = -sin(ab_rad) * 0.9877;
                ab_custom.at<double>(0, 2) = -2.8;
                ab_custom.at<double>(1, 0) = sin(ab_rad);
                ab_custom.at<double>(1, 1) = cos(ab_rad);
                ab_custom.at<double>(1, 2) = 0.0;

                cv::Mat bc_custom = cv::Mat::eye(3, 3, CV_64F);
                float bc_rad = 0.000f * CV_PI / 180.0f;
                bc_custom.at<double>(0, 0) = cos(bc_rad);
                bc_custom.at<double>(0, 1) = -sin(bc_rad);
                bc_custom.at<double>(0, 2) = 21.1;
                bc_custom.at<double>(1, 0) = sin(bc_rad);
                bc_custom.at<double>(1, 1) = cos(bc_rad);
                bc_custom.at<double>(1, 2) = 0.0;

                pipeline.SetBlendingMode(BlendingMode::FEATHERING);
                cv::Mat pano = pipeline.CreatePanoramaWithCustomTransforms(ab_custom, bc_custom);

                if (!pano.empty()) {
                    if (!panorama_writer_initialized) {
                        panorama_writer.open("output/panorama_result.avi",
                            cv::VideoWriter::fourcc('X','V','I','D'),
                            fps, pano.size());
                        panorama_writer_initialized = true;
                    }
                    panorama_writer.write(pano);
                }
            }
        }
        cap.release();
        writer.release();
    }

    if (panorama_writer_initialized) panorama_writer.release();

    cout << "✓ All videos saved." << endl;
    return 0;
}
