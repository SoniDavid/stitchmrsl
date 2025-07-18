// rtsp_stitcher_buffered.cpp
#include "stitching.hh"
#include <opencv2/opencv.hpp>
#include <csignal>
#include <atomic>
#include <thread>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <map>
#include <deque>

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

atomic<bool> stop_requested(false);
atomic<bool> cameras_stopped(false);
mutex cout_mutex;

void signal_handler(int signal) {
    if (signal == SIGINT) {
        stop_requested = true;
        cout << "\n[INFO] Ctrl+C received. Stopping capture and processing remaining frames...\n";
    }
}

void safe_cout(const string& message) {
    lock_guard<mutex> lock(cout_mutex);
    cout << message << endl;
}

// Structure to hold a frame with timestamp and sequence number
struct TimestampedFrame {
    cv::Mat frame;
    steady_clock::time_point timestamp;
    int camera_id;
    int sequence_number;
    
    TimestampedFrame() = default;
    TimestampedFrame(const cv::Mat& f, steady_clock::time_point t, int id, int seq) 
        : frame(f.clone()), timestamp(t), camera_id(id), sequence_number(seq) {}
};

// Thread-safe synchronized frame buffer
class SynchronizedFrameBuffer {
public:
    struct FrameSet {
        vector<TimestampedFrame> frames;
        steady_clock::time_point reference_time;
        bool complete;
        
        FrameSet() : complete(false) {
            frames.resize(3);
        }
    };
    
private:
    deque<FrameSet> complete_sets;
    map<int, deque<TimestampedFrame>> camera_buffers;
    mutex buffer_mutex;
    condition_variable buffer_cv;
    
    steady_clock::time_point sync_start_time;
    steady_clock::time_point latest_camera_start;
    map<int, steady_clock::time_point> camera_start_times;
    
    static const int MAX_BUFFER_SIZE = 200; // Increased buffer size
    static const int SYNC_TOLERANCE_MS = 50; // 50ms tolerance for frame synchronization
    
    void tryCreateFrameSets() {
        // Simple synchronization: just check if all cameras have frames
        while (true) {
            bool can_create_set = true;
            
            // Check if all cameras have at least one frame
            for (int cam = 0; cam < 3; cam++) {
                if (camera_buffers[cam].empty()) {
                    can_create_set = false;
                    break;
                }
            }
            
            if (!can_create_set) break;
            
            // Find the earliest timestamp among the first frames
            steady_clock::time_point ref_time = steady_clock::time_point::max();
            for (int cam = 0; cam < 3; cam++) {
                auto frame_time = camera_buffers[cam].front().timestamp;
                if (frame_time < ref_time) {
                    ref_time = frame_time;
                }
            }
            
            // Create frame set with frames closest to reference time
            FrameSet frame_set;
            frame_set.reference_time = ref_time;
            vector<int> selected_indices(3, 0);
            
            for (int cam = 0; cam < 3; cam++) {
                int best_idx = 0;
                auto best_diff = abs(duration_cast<milliseconds>(
                    camera_buffers[cam][0].timestamp - ref_time).count());
                
                // Look for the best matching frame within a reasonable window
                int search_limit = min(5, static_cast<int>(camera_buffers[cam].size()));
                for (int i = 1; i < search_limit; i++) {
                    auto diff = abs(duration_cast<milliseconds>(
                        camera_buffers[cam][i].timestamp - ref_time).count());
                    if (diff < best_diff) {
                        best_diff = diff;
                        best_idx = i;
                    }
                }
                
                selected_indices[cam] = best_idx;
                frame_set.frames[cam] = camera_buffers[cam][best_idx];
            }
            
            // Always add the frame set (remove strict synchronization requirement)
            frame_set.complete = true;
            complete_sets.push_back(frame_set);
            
            // Remove the oldest frame from each camera buffer
            for (int cam = 0; cam < 3; cam++) {
                if (!camera_buffers[cam].empty()) {
                    camera_buffers[cam].pop_front();
                }
            }
            
            // Limit complete sets buffer size
            if (complete_sets.size() > static_cast<size_t>(MAX_BUFFER_SIZE)) {
                complete_sets.pop_front();
            }
        }
    }
    
public:
    SynchronizedFrameBuffer() {
        for (int i = 0; i < 3; i++) {
            camera_buffers[i] = deque<TimestampedFrame>();
        }
    }
    
    void setCameraStartTime(int camera_id, steady_clock::time_point start_time) {
        lock_guard<mutex> lock(buffer_mutex);
        camera_start_times[camera_id] = start_time;
        
        // Update latest start time
        if (start_time > latest_camera_start) {
            latest_camera_start = start_time;
        }
        
        // If all cameras have reported their start times, set sync start
        if (camera_start_times.size() == 3) {
            sync_start_time = latest_camera_start + chrono::milliseconds(100); // Small buffer
        }
    }
    
    void addFrame(const TimestampedFrame& frame) {
        lock_guard<mutex> lock(buffer_mutex);
        
        // Add frame to buffer (remove sync start time restriction)
        camera_buffers[frame.camera_id].push_back(frame);
        
        // Limit individual camera buffer size
        if (camera_buffers[frame.camera_id].size() > static_cast<size_t>(MAX_BUFFER_SIZE)) {
            camera_buffers[frame.camera_id].pop_front();
        }
        
        // Try to create frame sets whenever we add a frame
        tryCreateFrameSets();
        
        buffer_cv.notify_one();
    }
    
    bool getFrameSet(FrameSet& frame_set, int timeout_ms = 1000) {
        unique_lock<mutex> lock(buffer_mutex);
        
        if (buffer_cv.wait_for(lock, chrono::milliseconds(timeout_ms), 
                              [this] { return !complete_sets.empty() || (stop_requested && cameras_stopped); })) {
            if (!complete_sets.empty()) {
                frame_set = complete_sets.front();
                complete_sets.pop_front();
                return true;
            }
        }
        return false;
    }
    
    size_t getQueueSize() {
        lock_guard<mutex> lock(buffer_mutex);
        return complete_sets.size();
    }
    
    size_t getTotalBufferedFrames() {
        lock_guard<mutex> lock(buffer_mutex);
        size_t total = complete_sets.size();
        for (const auto& buffer : camera_buffers) {
            total += buffer.second.size();
        }
        return total;
    }
    
    bool hasFrames() {
        lock_guard<mutex> lock(buffer_mutex);
        if (!complete_sets.empty()) return true;
        
        for (const auto& buffer : camera_buffers) {
            if (!buffer.second.empty()) return true;
        }
        return false;
    }
    
    void clear() {
        lock_guard<mutex> lock(buffer_mutex);
        complete_sets.clear();
        for (auto& buffer : camera_buffers) {
            buffer.second.clear();
        }
    }
    
    void finalizeProcessing() {
        lock_guard<mutex> lock(buffer_mutex);
        // Try to create final frame sets from remaining frames
        tryCreateFrameSets();
        buffer_cv.notify_all();
    }
};

// Camera capture class
class CameraCapture {
private:
    cv::VideoCapture cap;
    int camera_id;
    string rtsp_url;
    SynchronizedFrameBuffer& frame_buffer;
    thread capture_thread;
    steady_clock::time_point start_time;
    atomic<bool> is_running;
    int sequence_counter;
    cv::VideoWriter raw_video_writer;
    bool raw_writer_initialized;
    
    void captureLoop() {
        safe_cout("[CAM " + to_string(camera_id) + "] Starting capture from: " + rtsp_url);
        
        if (!cap.open(rtsp_url)) {
            safe_cout("[CAM " + to_string(camera_id) + "] Failed to open RTSP stream");
            return;
        }
        
        // Set minimal buffer to reduce latency
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 15.0;
        
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        
        safe_cout("[CAM " + to_string(camera_id) + "] Properties: " + 
                  to_string(width) + "x" + to_string(height) + " @ " + 
                  to_string(fps) + " FPS");
        
        // Record actual start time
        start_time = steady_clock::now();
        frame_buffer.setCameraStartTime(camera_id, start_time);
        
        sequence_counter = 0;
        int frame_count = 0;
        
        while (!stop_requested && is_running.load()) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                safe_cout("[CAM " + to_string(camera_id) + "] Failed to read frame or stream ended");
                break;
            }
            
            // Initialize raw video writer on first frame
            if (!raw_writer_initialized && !frame.empty()) {
                string raw_output_path = "output/camera_" + to_string(camera_id) + "_raw.mp4";
                cv::Size frame_size = frame.size();
                
                // Try multiple codecs for raw footage
                vector<pair<int, string>> codecs = {
                    {cv::VideoWriter::fourcc('m', 'p', '4', 'v'), "MP4V"},
                    {cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), "XVID"},
                    {cv::VideoWriter::fourcc('H', '2', '6', '4'), "H264"},
                    {cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), "MJPG"}
                };
                
                for (const auto& codec : codecs) {
                    raw_video_writer.open(raw_output_path, codec.first, fps, frame_size);
                    if (raw_video_writer.isOpened()) {
                        safe_cout("[CAM " + to_string(camera_id) + "] Raw video writer initialized with codec: " + codec.second);
                        raw_writer_initialized = true;
                        break;
                    }
                }
                
                if (!raw_writer_initialized) {
                    safe_cout("[CAM " + to_string(camera_id) + "] WARNING: Could not initialize raw video writer");
                }
            }
            
            // Save raw frame to video
            if (raw_writer_initialized) {
                raw_video_writer.write(frame);
            }
            
            auto timestamp = steady_clock::now();
            frame_buffer.addFrame(TimestampedFrame(frame, timestamp, camera_id, sequence_counter++));
            frame_count++;
            
            // Progress update every 5 seconds
            if (frame_count % (static_cast<int>(fps) * 5) == 0) {
                auto elapsed = duration_cast<seconds>(timestamp - start_time).count();
                safe_cout("[CAM " + to_string(camera_id) + "] Captured " + 
                          to_string(frame_count) + " frames (" + to_string(elapsed) + "s)");
            }
        }
        
        cap.release();
        if (raw_writer_initialized) {
            raw_video_writer.release();
        }
        safe_cout("[CAM " + to_string(camera_id) + "] Stopped capture. Total frames: " + 
                  to_string(frame_count));
    }
    
public:
    CameraCapture(int id, const string& url, SynchronizedFrameBuffer& buffer) 
        : camera_id(id), rtsp_url(url), frame_buffer(buffer), is_running(false), 
          sequence_counter(0), raw_writer_initialized(false) {}
    
    void start() {
        is_running.store(true);
        capture_thread = thread(&CameraCapture::captureLoop, this);
    }
    
    void stop() {
        is_running.store(false);
        if (capture_thread.joinable()) {
            capture_thread.join();
        }
    }
};

// Stitching processor class
class StitchingProcessor {
private:
    SynchronizedFrameBuffer& frame_buffer;
    StitchingPipeline pipeline;
    cv::VideoWriter video_writer;
    string output_path;
    double target_fps;
    bool writer_initialized;
    
    // Custom transformation matrices
    cv::Mat ab_custom, bc_custom;
    
    void initializeTransforms() {
        ab_custom = cv::Mat::eye(3, 3, CV_64F);
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
        
        bc_custom = cv::Mat::eye(3, 3, CV_64F);
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
    }
    
public:
    StitchingProcessor(SynchronizedFrameBuffer& buffer, const string& output, double fps = 15.0) 
        : frame_buffer(buffer), output_path(output), target_fps(fps), writer_initialized(false) {
        initializeTransforms();
        pipeline.SetBlendingMode(BlendingMode::AVERAGE);
    }
    
    bool initialize() {
        if (!pipeline.LoadIntrinsicsData("intrinsic.json") ||
            !pipeline.LoadExtrinsicsData("extrinsic.json")) {
            safe_cout("[ERROR] Failed to load calibration data.");
            return false;
        }
        return true;
    }
    
    void processFrames() {
        safe_cout("[INFO] Starting frame processing...");
        
        int processed_frames = 0;
        int successful_frames = 0;
        auto processing_start = steady_clock::now();
        
        while (!stop_requested || frame_buffer.hasFrames()) {
            SynchronizedFrameBuffer::FrameSet frame_set;
            
            // Get next synchronized frame set
            if (frame_buffer.getFrameSet(frame_set, 1000)) {
                auto process_start = steady_clock::now();
                
                // Clear pipeline cache
                pipeline.ClearCache();
                
                // Extract cv::Mat frames
                vector<cv::Mat> frames;
                for (const auto& ts_frame : frame_set.frames) {
                    frames.push_back(ts_frame.frame);
                }
                
                // Load frames into pipeline
                if (pipeline.LoadTestImagesFromMats(frames)) {
                    // Generate panorama
                    cv::Mat pano = pipeline.CreatePanoramaWithCustomTransforms(ab_custom, bc_custom);
                    
                    if (!pano.empty()) {
                        // Initialize video writer on first successful frame
                        if (!writer_initialized) {
                            cv::Size video_size = pano.size();
                            
                            // Try multiple codecs
                            vector<pair<int, string>> codecs = {
                                {cv::VideoWriter::fourcc('m', 'p', '4', 'v'), "MP4V"},
                                {cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), "XVID"},
                                {cv::VideoWriter::fourcc('H', '2', '6', '4'), "H264"},
                                {cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), "MJPG"}
                            };
                            
                            for (const auto& codec : codecs) {
                                video_writer.open(output_path, codec.first, target_fps, video_size);
                                if (video_writer.isOpened()) {
                                    safe_cout("[INFO] Video writer initialized with codec: " + codec.second);
                                    safe_cout("[INFO] Video size: " + to_string(video_size.width) + "x" + 
                                              to_string(video_size.height) + " @ " + to_string(target_fps) + " FPS");
                                    writer_initialized = true;
                                    break;
                                }
                            }
                            
                            if (!writer_initialized) {
                                safe_cout("[ERROR] Could not initialize video writer");
                                return;
                            }
                        }
                        
                        // Write frame to video
                        video_writer.write(pano);
                        successful_frames++;
                        
                        // Save debug frames periodically
                        if (processed_frames % 150 == 0) {
                            fs::create_directories("debug");
                            cv::imwrite("debug/pano_" + to_string(processed_frames) + ".png", pano);
                        }
                        
                        auto process_end = steady_clock::now();
                        auto process_time = duration_cast<milliseconds>(process_end - process_start).count();
                        
                        // Monitor processing performance
                        if (process_time > 66) { // More than one frame time at 15fps
                            safe_cout("[WARNING] Frame " + to_string(processed_frames) + 
                                      " took " + to_string(process_time) + "ms to process");
                        }
                    }
                }
                
                processed_frames++;
                
                // Progress update
                if (processed_frames % 50 == 0) {
                    auto elapsed = duration_cast<seconds>(steady_clock::now() - processing_start).count();
                    auto queue_size = frame_buffer.getQueueSize();
                    auto total_buffered = frame_buffer.getTotalBufferedFrames();
                    safe_cout("[INFO] Processed " + to_string(processed_frames) + 
                              " frames (" + to_string(successful_frames) + " successful) in " + 
                              to_string(elapsed) + "s, Queue: " + to_string(queue_size) + 
                              ", Total buffered: " + to_string(total_buffered));
                }
            } else {
                // No frames available
                if (stop_requested && cameras_stopped && !frame_buffer.hasFrames()) {
                    safe_cout("[INFO] No more frames to process, finishing...");
                    break;
                }
                
                // Show waiting message only if we're not stopping
                if (!stop_requested) {
                    safe_cout("[INFO] Waiting for frames...");
                }
            }
        }
        
        if (writer_initialized) {
            video_writer.release();
            safe_cout("[INFO] Video processing completed. Total frames: " + 
                      to_string(successful_frames));
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <rtsp_left> <rtsp_center> <rtsp_right>" << endl;
        return -1;
    }

    signal(SIGINT, signal_handler);
    
    fs::create_directories("output");
    fs::create_directories("debug");

    vector<string> rtsp_urls = { argv[1], argv[2], argv[3] };
    
    cout << "\n=== BUFFERED SYNCHRONIZED RTSP STITCHING ===" << endl;
    
    // Initialize synchronized frame buffer
    SynchronizedFrameBuffer sync_buffer;
    
    // Initialize processor
    StitchingProcessor processor(sync_buffer, "output/panorama_synchronized.mp4", 15.0);
    if (!processor.initialize()) {
        cerr << "[ERROR] Failed to initialize processor" << endl;
        return -1;
    }
    
    // Create camera capture instances
    vector<unique_ptr<CameraCapture>> cameras;
    for (int i = 0; i < 3; i++) {
        cameras.push_back(make_unique<CameraCapture>(i, rtsp_urls[i], sync_buffer));
    }
    
    // Start all cameras
    cout << "[INFO] Starting camera captures..." << endl;
    for (auto& camera : cameras) {
        camera->start();
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    
    // Start processing immediately
    thread processing_thread(&StitchingProcessor::processFrames, &processor);
    
    // Keep main thread alive
    while (!stop_requested) {
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    
    // Stop cameras first
    cout << "[INFO] Stopping cameras..." << endl;
    for (auto& camera : cameras) {
        camera->stop();
    }
    
    // Signal that cameras are stopped
    cameras_stopped = true;
    
    // Finalize any remaining frames in buffers
    sync_buffer.finalizeProcessing();
    
    // Wait for processing to complete
    if (processing_thread.joinable()) {
        cout << "[INFO] Processing remaining frames..." << endl;
        processing_thread.join();
    }
    
    cout << "\n=== PROCESSING COMPLETE ===" << endl;
    cout << "Final panorama: output/panorama_synchronized.mp4" << endl;
    cout << "Raw camera footage:" << endl;
    cout << "  Camera 0 (left): output/camera_0_raw.mp4" << endl;
    cout << "  Camera 1 (center): output/camera_1_raw.mp4" << endl;
    cout << "  Camera 2 (right): output/camera_2_raw.mp4" << endl;
    cout << "Debug frames: debug/" << endl;
    
    return 0;
}