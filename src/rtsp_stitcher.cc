#include "stitching.hh"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <rtsp_left> <rtsp_center> <rtsp_right>" << endl;
        return -1;
    }

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

    vector<cv::VideoCapture> caps(3);
    for (int i = 0; i < 3; ++i) {
        caps[i].open(rtsp_links[i]);
        if (!caps[i].isOpened()) {
            cerr << "Failed to open RTSP stream: " << rtsp_links[i] << endl;
            return -1;
        }
    }

    double fps = caps[1].get(cv::CAP_PROP_FPS);
    if (fps < 1.0) fps = 15.0;

    int width = caps[1].get(cv::CAP_PROP_FRAME_WIDTH);
    int height = caps[1].get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Size frame_size(width, height);

    vector<cv::VideoWriter> raw_writers(3);
    vector<cv::VideoWriter> rect_writers(3);

    for (int i = 0; i < 3; ++i) {
        raw_writers[i].open(raw_paths[i], cv::VideoWriter::fourcc('X','V','I','D'), fps, frame_size);
        rect_writers[i].open(rectified_paths[i], cv::VideoWriter::fourcc('X','V','I','D'), fps, frame_size);

        if (!raw_writers[i].isOpened() || !rect_writers[i].isOpened()) {
            cerr << "Failed to create writer for camera " << i << endl;
            return -1;
        }
    }

    cv::VideoWriter pano_writer;
    bool pano_writer_ready = false;

    StitchingPipeline pipeline;
    if (!pipeline.LoadIntrinsicsData("calibration/intrinsics.json") ||
        !pipeline.LoadExtrinsicsData("calibration/extrinsics.json")) {
        cerr << "Calibration loading failed!" << endl;
        return -1;
    }

    cout << "[INFO] Starting synchronized recording. Press ESC to stop..." << endl;

    while (true) {
        vector<cv::Mat> frames(3);
        bool all_read = true;

        for (int i = 0; i < 3; ++i) {
            if (!caps[i].read(frames[i]) || frames[i].empty()) {
                all_read = false;
                break;
            }
        }
        if (!all_read) break;

        for (int i = 0; i < 3; ++i) raw_writers[i].write(frames[i]);

        vector<cv::Mat> rectified(3);
        rectified[0] = pipeline.RectifyImageFisheye(frames[0], "izquierda", pipeline.GetCameraIntrinsicsByName("izquierda"));
        rectified[1] = pipeline.RectifyImageFisheye(frames[1], "central", pipeline.GetCameraIntrinsicsByName("central"));
        rectified[2] = pipeline.RectifyImageFisheye(frames[2], "derecha", pipeline.GetCameraIntrinsicsByName("derecha"));

        for (int i = 0; i < 3; ++i) rect_writers[i].write(rectified[i]);

        if (!pipeline.LoadTestImagesFromMats(rectified)) continue;

        // Manually tuned transforms
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
            if (!pano_writer_ready) {
                pano_writer.open("output/panorama_result.avi", cv::VideoWriter::fourcc('X','V','I','D'), fps, pano.size());
                pano_writer_ready = true;
            }
            pano_writer.write(pano);
        }

        if (cv::waitKey(1) == 27) break;  // ESC to stop
    }

    for (int i = 0; i < 3; ++i) {
        caps[i].release();
        raw_writers[i].release();
        rect_writers[i].release();
    }
    if (pano_writer_ready) pano_writer.release();

    cout << "\n[INFO] Recording and stitching complete. Videos saved to output/." << endl;
    return 0;
}
