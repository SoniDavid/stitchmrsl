// rtsp_stitcher.cc - Two-Phase Synchronized RTSP Stitching System
#include "synchronized_capture.hh"
#include "frame_synchronizer.hh"
#include "cuda_stitching.hh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <csignal>
#include <atomic>
#include <chrono>
#include <thread>
#include <iomanip>

namespace fs = std::filesystem;

// Global variables for signal handling
std::atomic<bool> g_stop_requested{false};
std::unique_ptr<SynchronizedCapture> g_capture_system;

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n[INFO] Ctrl+C received. Stopping capture..." << std::endl;
        g_stop_requested = true;
        if (g_capture_system) {
            g_capture_system->StopCapture();
        }
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <mode> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Modes:" << std::endl;
    std::cout << "  capture <rtsp1> <rtsp2> <rtsp3> [output_dir]" << std::endl;
    std::cout << "    - Capture synchronized frames from 3 RTSP streams" << std::endl;
    std::cout << "    - Default output_dir: ./output" << std::endl;
    std::cout << std::endl;
    std::cout << "  process <metadata_dir> [output_video]" << std::endl;
    std::cout << "    - Process captured frames into panorama video" << std::endl;
    std::cout << "    - Default output_video: ./panorama_result.mp4" << std::endl;
    std::cout << std::endl;
    std::cout << "  full <rtsp1> <rtsp2> <rtsp3> [output_dir] [output_video]" << std::endl;
    std::cout << "    - Full pipeline: capture then process" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " capture rtsp://192.168.1.10/stream rtsp://192.168.1.11/stream rtsp://192.168.1.12/stream" << std::endl;
    std::cout << "  " << program_name << " process ./output" << std::endl;
    std::cout << "  " << program_name << " full rtsp://cam1 rtsp://cam2 rtsp://cam3 ./data ./result.mp4" << std::endl;
}

bool load_metadata_files(const std::string& metadata_dir, 
                        std::vector<std::vector<FrameMetadata>>& camera_metadata) {
    
    std::vector<std::string> camera_names = {"cam1", "cam2", "cam3"};
    camera_metadata.resize(3);
    
    for (int i = 0; i < 3; ++i) {
        std::string metadata_file = metadata_dir + "/metadata/" + camera_names[i] + "_metadata.json";
        
        std::ifstream file(metadata_file);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Failed to open metadata file: " << metadata_file << std::endl;
            return false;
        }
        
        nlohmann::json metadata_json;
        file >> metadata_json;
        
        camera_metadata[i].clear();
        for (const auto& item : metadata_json) {
            camera_metadata[i].push_back(FrameMetadata::from_json(item));
        }
        
        std::cout << "[INFO] Loaded " << camera_metadata[i].size() 
                  << " frame metadata entries for " << camera_names[i] << std::endl;
    }
    
    return true;
}

bool create_panorama_video(const std::vector<SyncedFrameTriplet>& synchronized_frames,
                          const std::string& output_video_path) {
    
    if (synchronized_frames.empty()) {
        std::cerr << "[ERROR] No synchronized frames to process" << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Creating panorama video with " << synchronized_frames.size() 
              << " synchronized frames..." << std::endl;
    
    // Initialize CUDA stitching pipeline
    CUDAStitchingPipeline cuda_pipeline;
    
    // Estimate input size from first frame
    cv::Mat sample_frame = cv::imread(synchronized_frames[0].cam1_path);
    if (sample_frame.empty()) {
        std::cerr << "[ERROR] Failed to load sample frame: " << synchronized_frames[0].cam1_path << std::endl;
        return false;
    }
    
    cv::Size input_size = sample_frame.size();
    std::cout << "[INFO] Input frame size: " << input_size << std::endl;
    
    // Initialize CUDA pipeline
    if (!cuda_pipeline.Initialize(input_size)) {
        std::cerr << "[ERROR] Failed to initialize CUDA pipeline" << std::endl;
        return false;
    }
    
    // Set feathering blend mode by default for better overlap handling
    cuda_pipeline.SetBlendingMode(2); // 0=max, 1=average, 2=feathering
    cuda_pipeline.SetFeatheringRadius(50); // 50 pixel feathering radius
    
    // Load calibration data
    if (!cuda_pipeline.LoadCalibration("intrinsic.json", "extrinsic.json")) {
        std::cerr << "[ERROR] Failed to load calibration data" << std::endl;
        return false;
    }
    
    // Process first frame to get output size
    cv::Mat first_panorama = cuda_pipeline.ProcessFrameTriplet(
        synchronized_frames[0].cam1_path,
        synchronized_frames[0].cam2_path,
        synchronized_frames[0].cam3_path
    );
    
    if (first_panorama.empty()) {
        std::cerr << "[ERROR] Failed to process first frame" << std::endl;
        return false;
    }
    
    cv::Size output_size = first_panorama.size();
    std::cout << "[INFO] Output panorama size: " << output_size << std::endl;
    
    // Initialize video writer
    double fps = 24.0; // Target FPS
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter video_writer(output_video_path, fourcc, fps, output_size);
    
    if (!video_writer.isOpened()) {
        std::cerr << "[ERROR] Failed to open video writer: " << output_video_path << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Video writer initialized: " << output_video_path 
              << " (" << output_size << " @ " << fps << " FPS)" << std::endl;
    
    // Write first frame
    video_writer.write(first_panorama);
    
    // Process remaining frames
    auto process_start = std::chrono::steady_clock::now();
    size_t processed_frames = 1;
    
    for (size_t i = 1; i < synchronized_frames.size(); ++i) {
        if (g_stop_requested) {
            std::cout << "[INFO] Processing interrupted by user" << std::endl;
            break;
        }
        
        auto frame_start = std::chrono::steady_clock::now();
        
        cv::Mat panorama = cuda_pipeline.ProcessFrameTriplet(
            synchronized_frames[i].cam1_path,
            synchronized_frames[i].cam2_path,
            synchronized_frames[i].cam3_path
        );
        
        if (panorama.empty()) {
            std::cerr << "[WARNING] Failed to process frame " << i << ", skipping..." << std::endl;
            continue;
        }
        
        // Ensure consistent size
        if (panorama.size() != output_size) {
            cv::resize(panorama, panorama, output_size);
        }
        
        video_writer.write(panorama);
        processed_frames++;
        
        // Progress reporting
        if (i % 30 == 0 || i == synchronized_frames.size() - 1) {
            auto frame_end = std::chrono::steady_clock::now();
            double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                frame_end - frame_start).count();
            
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                frame_end - process_start).count();
            double progress = (double)i / synchronized_frames.size() * 100.0;
            double processing_fps = processed_frames / std::max(1.0, (double)elapsed);
            
            std::cout << "[INFO] Progress: " << std::fixed << std::setprecision(1) 
                      << progress << "% (" << i << "/" << synchronized_frames.size() 
                      << ") - Processing FPS: " << std::setprecision(1) << processing_fps
                      << " - Frame time: " << std::setprecision(0) << frame_time << "ms" << std::endl;
        }
    }
    
    video_writer.release();
    
    auto process_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::seconds>(
        process_end - process_start).count();
    double avg_fps = processed_frames / std::max(1.0, total_time);
    
    std::cout << "\n[INFO] ✅ Panorama video created successfully!" << std::endl;
    std::cout << "[INFO] Output: " << output_video_path << std::endl;
    std::cout << "[INFO] Processed frames: " << processed_frames << "/" << synchronized_frames.size() << std::endl;
    std::cout << "[INFO] Total processing time: " << std::fixed << std::setprecision(1) << total_time << "s" << std::endl;
    std::cout << "[INFO] Average processing FPS: " << std::setprecision(1) << avg_fps << std::endl;
    
    // Print CUDA pipeline statistics
    auto cuda_stats = cuda_pipeline.GetStats();
    std::cout << "[INFO] CUDA Pipeline Stats:" << std::endl;
    std::cout << "[INFO]   Average processing time: " << std::fixed << std::setprecision(2) 
              << cuda_stats.average_processing_time_ms << "ms per frame" << std::endl;
    std::cout << "[INFO]   GPU memory used: ~" << cuda_stats.gpu_memory_used_mb << "MB" << std::endl;
    
    return true;
}

int mode_capture(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "[ERROR] Insufficient arguments for capture mode" << std::endl;
        std::cerr << "Usage: " << argv[0] << " capture <rtsp1> <rtsp2> <rtsp3> [output_dir]" << std::endl;
        return -1;
    }
    
    std::vector<std::string> rtsp_urls = {argv[2], argv[3], argv[4]};
    std::string output_dir = (argc > 5) ? argv[5] : "./output";
    
    std::cout << "[INFO] === PHASE 1: SYNCHRONIZED CAPTURE ===" << std::endl;
    std::cout << "[INFO] RTSP URLs:" << std::endl;
    for (size_t i = 0; i < rtsp_urls.size(); ++i) {
        std::cout << "[INFO]   Camera " << (i+1) << ": " << rtsp_urls[i] << std::endl;
    }
    std::cout << "[INFO] Output directory: " << output_dir << std::endl;
    
    // Initialize capture system
    g_capture_system = std::make_unique<SynchronizedCapture>();
    
    // Start capture
    if (!g_capture_system->CaptureAllStreams(rtsp_urls, output_dir)) {
        std::cerr << "[ERROR] Failed to start capture" << std::endl;
        return -1;
    }
    
    std::cout << "[INFO] Capture started. Press Ctrl+C to stop..." << std::endl;
    
    // Wait for stop signal
    while (!g_stop_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Print statistics every 10 seconds
        static auto last_stats_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time).count() >= 10) {
            auto stats = g_capture_system->GetStats();
            std::cout << "[STATS] Frames captured: CAM1=" << stats.frames_captured[0] 
                      << ", CAM2=" << stats.frames_captured[1] 
                      << ", CAM3=" << stats.frames_captured[2] 
                      << " | Dropped: " << stats.dropped_frames 
                      << " | Duration: " << stats.total_duration.count() << "s" << std::endl;
            last_stats_time = now;
        }
    }
    
    // Save metadata
    std::cout << "[INFO] Saving metadata..." << std::endl;
    if (!g_capture_system->SaveMetadata(output_dir)) {
        std::cerr << "[ERROR] Failed to save metadata" << std::endl;
        return -1;
    }
    
    // Final statistics
    auto final_stats = g_capture_system->GetStats();
    std::cout << "\n[INFO] ✅ Capture completed successfully!" << std::endl;
    std::cout << "[INFO] Final Statistics:" << std::endl;
    std::cout << "[INFO]   Total duration: " << final_stats.total_duration.count() << "s" << std::endl;
    std::cout << "[INFO]   Frames captured: CAM1=" << final_stats.frames_captured[0] 
              << ", CAM2=" << final_stats.frames_captured[1] 
              << ", CAM3=" << final_stats.frames_captured[2] << std::endl;
    std::cout << "[INFO]   Average FPS: CAM1=" << std::fixed << std::setprecision(1) << final_stats.average_fps[0]
              << ", CAM2=" << final_stats.average_fps[1] 
              << ", CAM3=" << final_stats.average_fps[2] << std::endl;
    std::cout << "[INFO]   Dropped frames: " << final_stats.dropped_frames << std::endl;
    std::cout << "[INFO] Raw frames saved in: " << output_dir << "/raw_frames/" << std::endl;
    std::cout << "[INFO] Metadata saved in: " << output_dir << "/metadata/" << std::endl;
    
    return 0;
}

int mode_process(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "[ERROR] Insufficient arguments for process mode" << std::endl;
        std::cerr << "Usage: " << argv[0] << " process <metadata_dir> [output_video]" << std::endl;
        return -1;
    }
    
    std::string metadata_dir = argv[2];
    std::string output_video = (argc > 3) ? argv[3] : "./panorama_result.mp4";
    
    std::cout << "[INFO] === PHASE 2: SYNCHRONIZED PROCESSING ===" << std::endl;
    std::cout << "[INFO] Metadata directory: " << metadata_dir << std::endl;
    std::cout << "[INFO] Output video: " << output_video << std::endl;
    
    // Load metadata
    std::vector<std::vector<FrameMetadata>> camera_metadata;
    if (!load_metadata_files(metadata_dir, camera_metadata)) {
        std::cerr << "[ERROR] Failed to load metadata files" << std::endl;
        return -1;
    }
    
    // Synchronize frames
    std::cout << "[INFO] Synchronizing frames..." << std::endl;
    FrameSynchronizer synchronizer;
    
    // Configure synchronizer for post-processing (more lenient)
    synchronizer.SetSyncWindow(std::chrono::milliseconds(50)); // 50ms window
    synchronizer.SetMinSyncQuality(100.0); // 100ms max deviation
    
    auto synchronized_frames = synchronizer.FindOptimalSync(
        camera_metadata[0], camera_metadata[1], camera_metadata[2]);
    
    if (synchronized_frames.empty()) {
        std::cerr << "[ERROR] No synchronized frames found" << std::endl;
        return -1;
    }
    
    // Print synchronization statistics
    auto sync_stats = synchronizer.GetSyncStats();
    std::cout << "[INFO] Synchronization Results:" << std::endl;
    std::cout << "[INFO]   Synchronized triplets: " << sync_stats.total_triplets_found << std::endl;
    std::cout << "[INFO]   Perfect sync (<5ms): " << sync_stats.perfect_sync_triplets << std::endl;
    std::cout << "[INFO]   Good sync (<16.7ms): " << sync_stats.good_sync_triplets << std::endl;
    std::cout << "[INFO]   Average sync quality: " << std::fixed << std::setprecision(2) 
              << sync_stats.average_sync_quality << "ms" << std::endl;
    std::cout << "[INFO]   Effective FPS: " << std::setprecision(1) 
              << sync_stats.effective_fps << std::endl;
    
    // Create panorama video
    if (!create_panorama_video(synchronized_frames, output_video)) {
        std::cerr << "[ERROR] Failed to create panorama video" << std::endl;
        return -1;
    }
    
    return 0;
}

int mode_full(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "[ERROR] Insufficient arguments for full mode" << std::endl;
        std::cerr << "Usage: " << argv[0] << " full <rtsp1> <rtsp2> <rtsp3> [output_dir] [output_video]" << std::endl;
        return -1;
    }
    
    std::string output_dir = (argc > 5) ? argv[5] : "./output";
    std::string output_video = (argc > 6) ? argv[6] : "./panorama_result.mp4";
    
    std::cout << "[INFO] === FULL PIPELINE: CAPTURE + PROCESS ===" << std::endl;
    
    // Phase 1: Capture
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "PHASE 1: SYNCHRONIZED CAPTURE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Modify argv for capture mode
    char* capture_argv[] = {argv[0], (char*)"capture", argv[2], argv[3], argv[4], (char*)output_dir.c_str()};
    int capture_result = mode_capture(6, capture_argv);
    
    if (capture_result != 0) {
        std::cerr << "[ERROR] Capture phase failed" << std::endl;
        return capture_result;
    }
    
    // Phase 2: Process
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "PHASE 2: SYNCHRONIZED PROCESSING" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Reset stop signal for processing phase
    g_stop_requested = false;
    
    // Modify argv for process mode
    char* process_argv[] = {argv[0], (char*)"process", (char*)output_dir.c_str(), (char*)output_video.c_str()};
    int process_result = mode_process(4, process_argv);
    
    if (process_result != 0) {
        std::cerr << "[ERROR] Process phase failed" << std::endl;
        return process_result;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "[INFO] Raw frames: " << output_dir << "/raw_frames/" << std::endl;
    std::cout << "[INFO] Metadata: " << output_dir << "/metadata/" << std::endl;
    std::cout << "[INFO] Final video: " << output_video << std::endl;
    
    return 0;
}

int main(int argc, char** argv) {
    // Install signal handler
    std::signal(SIGINT, signal_handler);
    
    if (argc < 2) {
        print_usage(argv[0]);
        return -1;
    }
    
    std::string mode = argv[1];
    
    std::cout << "[INFO] RTSP Stitcher v2.0 - Two-Phase Synchronized Processing" << std::endl;
    std::cout << "[INFO] Mode: " << mode << std::endl;
    
    if (mode == "capture") {
        return mode_capture(argc, argv);
    } else if (mode == "process") {
        return mode_process(argc, argv);
    } else if (mode == "full") {
        return mode_full(argc, argv);
    } else {
        std::cerr << "[ERROR] Unknown mode: " << mode << std::endl;
        print_usage(argv[0]);
        return -1;
    }
}