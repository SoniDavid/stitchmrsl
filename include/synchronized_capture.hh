#ifndef SYNCHRONIZED_CAPTURE_HH
#define SYNCHRONIZED_CAPTURE_HH

#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct FrameMetadata {
    std::chrono::high_resolution_clock::time_point timestamp;
    uint64_t sequence_number;
    std::string frame_path;
    int camera_id;
    cv::Size frame_size;
    bool is_valid;
    
    // Serialize to JSON
    json to_json() const;
    static FrameMetadata from_json(const json& j);
};

struct WriteTask {
    cv::Mat frame;
    std::string filename;
    std::chrono::high_resolution_clock::time_point timestamp;
    int camera_id;
    uint64_t sequence_number;
    
    // Default constructor
    WriteTask() = default;
    
    // Parameterized constructor
    WriteTask(const cv::Mat& f, const std::string& fn, 
              std::chrono::high_resolution_clock::time_point ts,
              int cam_id, uint64_t seq)
        : frame(f.clone()), filename(fn), timestamp(ts), 
          camera_id(cam_id), sequence_number(seq) {}
};

class SynchronizedCapture {
public:
    SynchronizedCapture();
    ~SynchronizedCapture();
    
    // Main capture function
    bool CaptureAllStreams(const std::vector<std::string>& rtsp_urls, 
                          const std::string& output_dir);
    
    // Stop capture gracefully
    void StopCapture();
    
    // Save metadata to JSON files
    bool SaveMetadata(const std::string& output_dir);
    
    // Get capture statistics
    struct CaptureStats {
        std::vector<size_t> frames_captured;
        std::chrono::seconds total_duration;
        std::vector<double> average_fps;
        size_t dropped_frames;
    };
    CaptureStats GetStats() const;

private:
    // Per-camera capture thread
    void CaptureCamera(int camera_id, const std::string& rtsp_url, 
                      const std::string& output_dir);
    
    // Write frames asynchronously
    void AsyncFrameWriter();
    
    // Thread management
    std::vector<std::thread> capture_threads_;
    std::vector<std::thread> writer_threads_;
    
    // Synchronization
    std::atomic<bool> stop_requested_;
    std::atomic<bool> capture_active_;
    
    // Frame metadata storage
    std::vector<std::vector<FrameMetadata>> camera_metadata_;
    std::mutex metadata_mutex_;
    
    // Asynchronous writing
    std::queue<WriteTask> write_queue_;
    std::mutex write_queue_mutex_;
    std::condition_variable write_queue_cv_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    std::vector<size_t> frames_captured_;
    std::chrono::high_resolution_clock::time_point capture_start_time_;
    size_t dropped_frames_;
    
    // Configuration
    const unsigned int NUM_WRITER_THREADS;
    static constexpr size_t MAX_WRITE_QUEUE_SIZE = 1000;
    static constexpr int CAPTURE_TIMEOUT_MS = 5000;
};

#endif // SYNCHRONIZED_CAPTURE_HH