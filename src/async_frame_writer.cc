#include "async_frame_writer.hh"
#include <iostream>
#include <filesystem>
#include <numeric>

AsyncFrameWriter::AsyncFrameWriter() 
    : stop_requested_(false), running_(false), max_queue_size_(DEFAULT_MAX_QUEUE_SIZE),
      frames_written_(0), frames_dropped_(0), total_bytes_written_(0) {
    
    // Set default PNG compression parameters
    compression_params_ = {cv::IMWRITE_PNG_COMPRESSION, 6}; // Moderate compression
}

AsyncFrameWriter::~AsyncFrameWriter() {
    Stop();
}

bool AsyncFrameWriter::Start(size_t max_queue_size) {
    if (running_) {
        std::cerr << "[WRITER] Already running" << std::endl;
        return false;
    }
    
    max_queue_size_ = max_queue_size;
    stop_requested_ = false;
    running_ = true;
    start_time_ = std::chrono::high_resolution_clock::now();
    
    // Clear statistics
    frames_written_ = 0;
    frames_dropped_ = 0;
    total_bytes_written_ = 0;
    write_times_.clear();
    
    // Start writer thread
    writer_thread_ = std::thread(&AsyncFrameWriter::WriterThreadFunc, this);
    
    std::cout << "[WRITER] AsyncFrameWriter started with queue size: " << max_queue_size_ << std::endl;
    return true;
}

void AsyncFrameWriter::Stop() {
    if (!running_) {
        return;
    }
    
    std::cout << "[WRITER] Stopping AsyncFrameWriter..." << std::endl;
    stop_requested_ = true;
    
    // Wake up writer thread
    queue_cv_.notify_all();
    
    // Wait for writer thread to finish
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
    
    running_ = false;
    
    // Print final statistics
    auto stats = GetStats();
    std::cout << "[WRITER] Final stats - Frames written: " << stats.frames_written 
              << ", Dropped: " << stats.frames_dropped 
              << ", Avg write time: " << std::fixed << std::setprecision(2) 
              << stats.average_write_time_ms << "ms" << std::endl;
}

bool AsyncFrameWriter::QueueFrame(const cv::Mat& frame, const std::string& filename,
                                 std::chrono::high_resolution_clock::time_point timestamp,
                                 int camera_id, uint64_t sequence_number) {
    if (!running_) {
        return false;
    }
    
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Check if queue is full
    if (write_queue_.size() >= max_queue_size_) {
        frames_dropped_++;
        return false;
    }
    
    // Add task to queue
    write_queue_.emplace(frame, filename, timestamp, camera_id, sequence_number);
    
    // Notify writer thread
    queue_cv_.notify_one();
    
    return true;
}

bool AsyncFrameWriter::IsQueueFull() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return write_queue_.size() >= max_queue_size_;
}

size_t AsyncFrameWriter::GetQueueSize() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return write_queue_.size();
}

void AsyncFrameWriter::WriterThreadFunc() {
    std::cout << "[WRITER] Writer thread started" << std::endl;
    
    while (running_ || !write_queue_.empty()) {
        ProcessWriteQueue();
        
        // If no more work and stop requested, exit
        if (stop_requested_ && write_queue_.empty()) {
            break;
        }
        
        // Small sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    std::cout << "[WRITER] Writer thread finished" << std::endl;
}

void AsyncFrameWriter::ProcessWriteQueue() {
    AsyncWriteTask task;
    bool has_task = false;
    
    // Get task from queue
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for work or stop signal
        queue_cv_.wait_for(lock, QUEUE_TIMEOUT, [this] {
            return !write_queue_.empty() || stop_requested_;
        });
        
        if (!write_queue_.empty()) {
            task = std::move(write_queue_.front());
            write_queue_.pop();
            has_task = true;
        }
    }
    
    if (has_task) {
        WriteFrame(task);
    }
}

bool AsyncFrameWriter::WriteFrame(const AsyncWriteTask& task) {
    auto write_start = std::chrono::high_resolution_clock::now();
    
    // Ensure directory exists
    std::filesystem::path file_path(task.filename);
    std::filesystem::create_directories(file_path.parent_path());
    
    // Write frame to disk
    bool success = cv::imwrite(task.filename, task.frame, compression_params_);
    
    auto write_end = std::chrono::high_resolution_clock::now();
    double write_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        write_end - write_start).count();
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        
        if (success) {
            frames_written_++;
            
            // Estimate file size (rough approximation)
            size_t estimated_size = task.frame.total() * task.frame.elemSize();
            total_bytes_written_ += estimated_size;
            
            // Store write time for average calculation
            write_times_.push_back(write_time);
            if (write_times_.size() > MAX_WRITE_TIME_SAMPLES) {
                write_times_.erase(write_times_.begin());
            }
        } else {
            frames_dropped_++;
            std::cerr << "[WRITER] Failed to write frame: " << task.filename << std::endl;
        }
    }
    
    return success;
}

AsyncFrameWriter::WriterStats AsyncFrameWriter::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    WriterStats stats;
    stats.frames_written = frames_written_;
    stats.frames_dropped = frames_dropped_;
    stats.current_queue_size = GetQueueSize();
    stats.total_bytes_written = total_bytes_written_;
    stats.uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - start_time_);
    
    // Calculate average write time
    if (!write_times_.empty()) {
        double sum = std::accumulate(write_times_.begin(), write_times_.end(), 0.0);
        stats.average_write_time_ms = sum / write_times_.size();
    } else {
        stats.average_write_time_ms = 0.0;
    }
    
    return stats;
}

void AsyncFrameWriter::SetCompressionParams(const std::vector<int>& params) {
    compression_params_ = params;
    std::cout << "[WRITER] Compression parameters updated" << std::endl;
}