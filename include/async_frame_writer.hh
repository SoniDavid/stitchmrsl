#ifndef ASYNC_FRAME_WRITER_HH
#define ASYNC_FRAME_WRITER_HH

#include <opencv2/opencv.hpp>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

struct AsyncWriteTask {
    cv::Mat frame;
    std::string filename;
    std::chrono::high_resolution_clock::time_point timestamp;
    int camera_id;
    uint64_t sequence_number;
    
    AsyncWriteTask() = default;
    AsyncWriteTask(const cv::Mat& f, const std::string& fn, 
                   std::chrono::high_resolution_clock::time_point ts,
                   int cam_id, uint64_t seq)
        : frame(f.clone()), filename(fn), timestamp(ts), 
          camera_id(cam_id), sequence_number(seq) {}
};

class AsyncFrameWriter {
public:
    AsyncFrameWriter();
    ~AsyncFrameWriter();
    
    // Start the writer thread
    bool Start(size_t max_queue_size = 100);
    
    // Stop the writer thread gracefully
    void Stop();
    
    // Queue a frame for writing
    bool QueueFrame(const cv::Mat& frame, const std::string& filename,
                   std::chrono::high_resolution_clock::time_point timestamp,
                   int camera_id, uint64_t sequence_number);
    
    // Check if queue is full
    bool IsQueueFull() const;
    
    // Get current queue size
    size_t GetQueueSize() const;
    
    // Get writing statistics
    struct WriterStats {
        size_t frames_written;
        size_t frames_dropped;
        size_t current_queue_size;
        double average_write_time_ms;
        size_t total_bytes_written;
        std::chrono::seconds uptime;
    };
    WriterStats GetStats() const;
    
    // Set compression parameters
    void SetCompressionParams(const std::vector<int>& params);

private:
    // Writer thread function
    void WriterThreadFunc();
    
    // Process write queue
    void ProcessWriteQueue();
    
    // Write single frame
    bool WriteFrame(const AsyncWriteTask& task);
    
    // Thread management
    std::thread writer_thread_;
    std::atomic<bool> stop_requested_;
    std::atomic<bool> running_;
    
    // Queue management
    std::queue<AsyncWriteTask> write_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    size_t max_queue_size_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    size_t frames_written_;
    size_t frames_dropped_;
    size_t total_bytes_written_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<double> write_times_;
    
    // Compression parameters
    std::vector<int> compression_params_;
    
    // Constants
    static constexpr size_t DEFAULT_MAX_QUEUE_SIZE = 100;
    static constexpr std::chrono::milliseconds QUEUE_TIMEOUT{100};
    static constexpr size_t MAX_WRITE_TIME_SAMPLES = 1000;
};

#endif // ASYNC_FRAME_WRITER_HH