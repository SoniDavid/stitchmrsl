#include "synchronized_capture.hh"
#include "async_frame_writer.hh"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

namespace fs = std::filesystem;

// FrameMetadata JSON serialization
json FrameMetadata::to_json() const {
    return json{
        {"timestamp", std::chrono::duration_cast<std::chrono::nanoseconds>(
            timestamp.time_since_epoch()).count()},
        {"sequence_number", sequence_number},
        {"frame_path", frame_path},
        {"camera_id", camera_id},
        {"frame_width", frame_size.width},
        {"frame_height", frame_size.height},
        {"is_valid", is_valid}
    };
}

FrameMetadata FrameMetadata::from_json(const json& j) {
    FrameMetadata meta;
    meta.timestamp = std::chrono::high_resolution_clock::time_point(
        std::chrono::nanoseconds(j["timestamp"].get<int64_t>()));
    meta.sequence_number = j["sequence_number"];
    meta.frame_path = j["frame_path"];
    meta.camera_id = j["camera_id"];
    meta.frame_size.width = j["frame_width"];
    meta.frame_size.height = j["frame_height"];
    meta.is_valid = j["is_valid"];
    return meta;
}

SynchronizedCapture::SynchronizedCapture() 
    : stop_requested_(false), capture_active_(false), dropped_frames_(0), NUM_WRITER_THREADS(std::thread::hardware_concurrency()) {
    camera_metadata_.resize(3);
    frames_captured_.resize(3, 0);
}

SynchronizedCapture::~SynchronizedCapture() {
    StopCapture();
}

bool SynchronizedCapture::CaptureAllStreams(const std::vector<std::string>& rtsp_urls, 
                                          const std::string& output_dir) {
    if (rtsp_urls.size() != 3) {
        std::cerr << "[ERROR] Expected 3 RTSP URLs, got " << rtsp_urls.size() << std::endl;
        return false;
    }
    
    // Create output directories
    std::vector<std::string> camera_dirs = {"cam1", "cam2", "cam3"};
    for (const auto& dir : camera_dirs) {
        fs::create_directories(output_dir + "/raw_frames/" + dir);
    }
    fs::create_directories(output_dir + "/metadata");
    
    std::cout << "[INFO] Starting synchronized capture..." << std::endl;
    std::cout << "[INFO] Output directory: " << output_dir << std::endl;
    
    // Reset state
    stop_requested_ = false;
    capture_active_ = true;
    capture_start_time_ = std::chrono::high_resolution_clock::now();
    
    // Clear previous metadata
    for (auto& meta : camera_metadata_) {
        meta.clear();
    }
    std::fill(frames_captured_.begin(), frames_captured_.end(), 0);
    dropped_frames_ = 0;
    
    // Start writer threads
    writer_threads_.clear();
    std::cout << "[INFO] Starting " << NUM_WRITER_THREADS << " writer threads..." << std::endl;
    for (unsigned int i = 0; i < NUM_WRITER_THREADS; ++i) {
        writer_threads_.emplace_back(&SynchronizedCapture::AsyncFrameWriter, this);
    }
    
    // Start capture threads
    capture_threads_.clear();
    for (int i = 0; i < 3; ++i) {
        capture_threads_.emplace_back(&SynchronizedCapture::CaptureCamera, 
                                    this, i, rtsp_urls[i], output_dir);
        std::cout << "[INFO] Started capture thread for camera " << i 
                  << " (" << rtsp_urls[i] << ")" << std::endl;
    }
    
    return true;
}

void SynchronizedCapture::StopCapture() {
    if (!capture_active_) {
        return;
    }
    
    std::cout << "[INFO] Stopping capture..." << std::endl;
    stop_requested_ = true;
    
    // Wait for capture threads to finish
    for (auto& thread : capture_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    capture_threads_.clear();
    
    // Signal writer threads to stop
    {
        std::lock_guard<std::mutex> lock(write_queue_mutex_);
        write_queue_cv_.notify_all();
    }
    
    // Wait for writer threads to finish
    for (auto& thread : writer_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    writer_threads_.clear();
    
    capture_active_ = false;
    std::cout << "[INFO] Capture stopped successfully" << std::endl;
}

void SynchronizedCapture::CaptureCamera(int camera_id, const std::string& rtsp_url, 
                                       const std::string& output_dir) {
    cv::VideoCapture cap(rtsp_url);
    if (!cap.isOpened()) {
        std::cerr << "[CAM " << camera_id << "] Failed to open: " << rtsp_url << std::endl;
        return;
    }
    
    // Configure capture
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1); // Minimize buffering for real-time
    cap.set(cv::CAP_PROP_FPS, 30); // Request 30 FPS
    
    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "[CAM " << camera_id << "] Properties: " << width << "x" << height 
              << " @ " << fps << " FPS" << std::endl;
    
    uint64_t sequence_number = 0;
    std::string camera_dir = output_dir + "/raw_frames/cam" + std::to_string(camera_id + 1);
    
    auto last_fps_report = std::chrono::steady_clock::now();
    size_t frames_since_report = 0;
    
    while (!stop_requested_) {
        cv::Mat frame;
        auto capture_time = std::chrono::high_resolution_clock::now();
        
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "[CAM " << camera_id << "] Failed to read frame or stream ended" << std::endl;
            break;
        }
        
        // Create frame metadata
        FrameMetadata metadata;
        metadata.timestamp = capture_time;
        metadata.sequence_number = sequence_number++;
        metadata.camera_id = camera_id;
        metadata.frame_size = frame.size();
        metadata.is_valid = true;
        
        // Generate filename
        std::stringstream ss;
        ss << camera_dir << "/" << std::setfill('0') << std::setw(8) 
           << metadata.sequence_number << ".png";
        metadata.frame_path = ss.str();
        
        // Queue frame for async writing
        {
            std::lock_guard<std::mutex> lock(write_queue_mutex_);
            if (write_queue_.size() < MAX_WRITE_QUEUE_SIZE) {
                write_queue_.emplace(frame, metadata.frame_path, capture_time, 
                                   camera_id, sequence_number - 1);
                write_queue_cv_.notify_one();
            } else {
                dropped_frames_++;
                std::cerr << "[CAM " << camera_id << "] Dropped frame (queue full)" << std::endl;
                continue; // Skip metadata storage for dropped frames
            }
        }
        
        // Store metadata
        {
            std::lock_guard<std::mutex> lock(metadata_mutex_);
            camera_metadata_[camera_id].push_back(metadata);
        }
        
        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            frames_captured_[camera_id]++;
        }
        
        frames_since_report++;
        
        // Report FPS every 5 seconds
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_fps_report).count() >= 5) {
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_fps_report).count() / 1000.0;
            double current_fps = frames_since_report / elapsed;
            
            std::cout << "[CAM " << camera_id << "] Captured " << frames_since_report 
                      << " frames in " << std::fixed << std::setprecision(1) << elapsed 
                      << "s (FPS: " << std::setprecision(1) << current_fps << ")" << std::endl;
            
            frames_since_report = 0;
            last_fps_report = now;
        }
    }
    
    cap.release();
    std::cout << "[CAM " << camera_id << "] Capture thread finished. Total frames: " 
              << frames_captured_[camera_id] << std::endl;
}

void SynchronizedCapture::AsyncFrameWriter() {
    std::cout << "[WRITER] Frame writer thread started" << std::endl;
    
    while (!stop_requested_ || !write_queue_.empty()) {
        WriteTask task;
        bool has_task = false;
        
        // Get task from queue
        {
            std::unique_lock<std::mutex> lock(write_queue_mutex_);
            write_queue_cv_.wait_for(lock, std::chrono::milliseconds(100), 
                                   [this] { return !write_queue_.empty() || stop_requested_; });
            
            if (!write_queue_.empty()) {
                task = write_queue_.front();
                write_queue_.pop();
                has_task = true;
            }
        }
        
        if (has_task) {
            // Write frame to disk with no compression for speed
            if (!cv::imwrite(task.filename, task.frame, {cv::IMWRITE_PNG_COMPRESSION, 0})) {
                std::cerr << "[WRITER] Failed to write frame: " << task.filename << std::endl;
            }
        }
    }
    
    std::cout << "[WRITER] Frame writer thread finished" << std::endl;
}

bool SynchronizedCapture::SaveMetadata(const std::string& output_dir) {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    
    std::vector<std::string> camera_names = {"cam1", "cam2", "cam3"};
    
    for (int i = 0; i < 3; ++i) {
        std::string metadata_file = output_dir + "/metadata/" + camera_names[i] + "_metadata.json";
        
        json metadata_json = json::array();
        for (const auto& meta : camera_metadata_[i]) {
            metadata_json.push_back(meta.to_json());
        }
        
        std::ofstream file(metadata_file);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Failed to create metadata file: " << metadata_file << std::endl;
            return false;
        }
        
        file << metadata_json.dump(2);
        file.close();
        
        std::cout << "[INFO] Saved metadata for camera " << i << ": " 
                  << camera_metadata_[i].size() << " frames" << std::endl;
    }
    
    // Save capture summary
    json summary;
    summary["capture_start_time"] = std::chrono::duration_cast<std::chrono::nanoseconds>(
        capture_start_time_.time_since_epoch()).count();
    summary["total_duration_seconds"] = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - capture_start_time_).count();
    
    for (int i = 0; i < 3; ++i) {
        summary["frames_captured"][camera_names[i]] = frames_captured_[i];
    }
    summary["dropped_frames"] = dropped_frames_;
    
    std::ofstream summary_file(output_dir + "/metadata/capture_summary.json");
    summary_file << summary.dump(2);
    summary_file.close();
    
    return true;
}

SynchronizedCapture::CaptureStats SynchronizedCapture::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    CaptureStats stats;
    stats.frames_captured = frames_captured_;
    stats.dropped_frames = dropped_frames_;
    stats.total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - capture_start_time_);
    
    // Calculate average FPS
    stats.average_fps.resize(3);
    double duration_seconds = stats.total_duration.count();
    if (duration_seconds > 0) {
        for (int i = 0; i < 3; ++i) {
            stats.average_fps[i] = frames_captured_[i] / duration_seconds;
        }
    }
    
    return stats;
}