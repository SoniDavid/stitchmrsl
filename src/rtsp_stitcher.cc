// Enhanced rtsp_stitcher.cpp with timestamp-based synchronization
#include "stitching.hh"
#include <opencv2/opencv.hpp>
#include <csignal>
#include <atomic>
#include <thread>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <queue>
#include <map>
#include <vector>
#include <algorithm>

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

atomic<bool> stop_requested(false);

// Structure to hold frame with timestamp
struct TimestampedFrame {
    cv::Mat frame;
    steady_clock::time_point timestamp;
    int64_t timestamp_ms;
    int frame_index;
    int camera_id;
};

// Thread-safe frame buffer for each camera
class FrameBuffer {
private:
    queue<TimestampedFrame> buffer;
    mutable mutex buffer_mutex;
    const size_t max_size;
    
public:
    FrameBuffer(size_t max_buffer_size = 100) : max_size(max_buffer_size) {}
    
    void Push(const TimestampedFrame& frame) {
        lock_guard<mutex> lock(buffer_mutex);
        buffer.push(frame);
        
        // Remove old frames if buffer is full
        while (buffer.size() > max_size) {
            buffer.pop();
        }
    }
    
    bool Pop(TimestampedFrame& frame) {
        lock_guard<mutex> lock(buffer_mutex);
        if (buffer.empty()) return false;
        
        frame = buffer.front();
        buffer.pop();
        return true;
    }
    
    bool GetClosestFrame(int64_t target_timestamp_ms, TimestampedFrame& frame, int64_t tolerance_ms = 100) {
        lock_guard<mutex> lock(buffer_mutex);
        if (buffer.empty()) return false;
        
        // Find the frame with timestamp closest to target
        TimestampedFrame closest_frame;
        int64_t min_diff = LLONG_MAX;
        bool found = false;
        
        queue<TimestampedFrame> temp_buffer = buffer;
        
        while (!temp_buffer.empty()) {
            TimestampedFrame current = temp_buffer.front();
            temp_buffer.pop();
            
            int64_t diff = abs(current.timestamp_ms - target_timestamp_ms);
            if (diff <= tolerance_ms && diff < min_diff) {
                min_diff = diff;
                closest_frame = current;
                found = true;
            }
        }
        
        if (found) {
            frame = closest_frame;
            return true;
        }
        return false;
    }
    
    size_t Size() const {
        lock_guard<mutex> lock(buffer_mutex);
        return buffer.size();
    }
    
    void Clear() {
        lock_guard<mutex> lock(buffer_mutex);
        while (!buffer.empty()) {
            buffer.pop();
        }
    }
};

// Process frames and create panorama video (from original code)
bool ProcessFramesToPanorama(const string& frames_base_dir, const string& output_video_path, 
                           StitchingPipeline& pipeline, double fps = 15.0) {
    
    // Find the minimum number of frames across all cameras
    vector<string> camera_names = {"left", "central", "right"};
    vector<string> frame_dirs = {
        frames_base_dir + "/left",
        frames_base_dir + "/central", 
        frames_base_dir + "/right"
    };
    
    int min_frames = INT_MAX;
    for (const auto& dir : frame_dirs) {
        int count = 0;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".png") count++;
        }
        min_frames = min(min_frames, count);
        cout << "[INFO] Found " << count << " frames in " << dir << endl;
    }
    
    if (min_frames == 0) {
        cerr << "[ERROR] No frames found in one or more directories" << endl;
        return false;
    }
    
    cout << "[INFO] Processing " << min_frames << " synchronized frames" << endl;
    
    // Setup custom transformation matrices (same as original)
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

    pipeline.SetBlendingMode(BlendingMode::AVERAGE);
    
    cv::VideoWriter pano_writer;
    bool writer_initialized = false;
    cv::Size video_size;
    int successful_frames = 0;
    
    for (int frame_idx = 0; frame_idx < min_frames; frame_idx++) {
        cout << "[INFO] Processing frame " << frame_idx << "/" << min_frames << endl;
        
        // CRITICAL: Clear pipeline cache before each frame
        pipeline.ClearCache();
        
        // Load synchronized frames
        vector<cv::Mat> frames(3);
        vector<string> frame_paths(3);
        bool frames_loaded = true;
        
        for (int cam = 0; cam < 3; cam++) {
            stringstream ss;
            ss << frame_dirs[cam] << "/" << camera_names[cam] << "_" << setfill('0') << setw(6) << frame_idx << ".png";
            frame_paths[cam] = ss.str();
            
            // Check if file exists first
            if (!fs::exists(frame_paths[cam])) {
                cerr << "[ERROR] Frame file does not exist: " << frame_paths[cam] << endl;
                frames_loaded = false;
                break;
            }
            
            frames[cam] = cv::imread(frame_paths[cam], cv::IMREAD_COLOR);
            if (frames[cam].empty()) {
                cerr << "[ERROR] Failed to load frame: " << frame_paths[cam] << endl;
                frames_loaded = false;
                break;
            }
            
            // Verify frame dimensions
            if (frames[cam].rows == 0 || frames[cam].cols == 0) {
                cerr << "[ERROR] Invalid frame dimensions for: " << frame_paths[cam] << endl;
                frames_loaded = false;
                break;
            }
        }
        
        if (!frames_loaded) {
            cerr << "[ERROR] Skipping frame " << frame_idx << " due to loading issues" << endl;
            continue;
        }
        
        // Load fresh frames into pipeline
        if (!pipeline.LoadTestImagesFromMats(frames)) {
            cerr << "[ERROR] Failed to load frames into pipeline for frame " << frame_idx << endl;
            continue;
        }
        
        // Generate panorama with fresh data
        cv::Mat pano = pipeline.CreatePanoramaWithCustomTransforms(ab_custom, bc_custom);
        if (pano.empty()) {
            cerr << "[WARNING] Empty panorama at frame " << frame_idx << endl;
            continue;
        }
        
        // Initialize video writer on first successful frame
        if (!writer_initialized) {
            video_size = pano.size();
            
            // Try multiple codecs in order of preference
            vector<pair<int, string>> codecs = {
                {cv::VideoWriter::fourcc('m', 'p', '4', 'v'), "MP4V"},
                {cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), "XVID"},
                {cv::VideoWriter::fourcc('H', '2', '6', '4'), "H264"},
                {cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), "MJPG"}
            };
            
            for (const auto& codec : codecs) {
                pano_writer.open(output_video_path, codec.first, fps, video_size);
                if (pano_writer.isOpened()) {
                    cout << "[INFO] Successfully initialized video writer with codec: " << codec.second << endl;
                    cout << "[INFO] Video size: " << video_size << " @ " << fps << " FPS" << endl;
                    writer_initialized = true;
                    break;
                }
                cout << "[WARNING] Failed to initialize with codec: " << codec.second << endl;
            }
            
            if (!writer_initialized) {
                cerr << "[ERROR] Could not initialize video writer with any codec" << endl;
                return false;
            }
        }
        
        // Ensure panorama size matches video size
        if (pano.size() != video_size) {
            cerr << "[WARNING] Frame " << frame_idx << " size mismatch. Expected: " << video_size 
                 << ", Got: " << pano.size() << ". Resizing..." << endl;
            cv::resize(pano, pano, video_size);
        }
        
        // Write frame to video
        pano_writer.write(pano);
        successful_frames++;
        
        // Progress update
        if (frame_idx % 30 == 0) {
            cout << "[INFO] Processed frame " << frame_idx << "/" << min_frames 
                 << " (successful: " << successful_frames << ")" << endl;
        }
    }
    
    if (writer_initialized) {
        pano_writer.release();
        cout << "[INFO] Panorama video saved: " << output_video_path << endl;
        cout << "[INFO] Total successful frames written: " << successful_frames << "/" << min_frames << endl;
    }
    
    return true;
}

void signal_handler(int signal) {
    if (signal == SIGINT) {
        stop_requested = true;
        cout << "\n[INFO] Ctrl+C received. Finishing current recording before exit...\n";
    }
}

// Enhanced capture function with timestamp recording
void CaptureRTSPWithTimestamps(const string& url, int cam_id, FrameBuffer& frame_buffer) {
    cv::VideoCapture cap(url);
    if (!cap.isOpened()) {
        cerr << "[CAM " << cam_id << "] Failed to open: " << url << endl;
        return;
    }

    // Set buffer size to minimize latency
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    
    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 15.0; // Default fallback
    
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    cout << "[CAM " << cam_id << "] Properties: " << width << "x" << height << " @ " << fps << " FPS" << endl;
    cout << "[CAM " << cam_id << "] Started capturing with timestamp sync" << endl;
    
    int frame_count = 0;
    auto start_time = steady_clock::now();
    
    while (!stop_requested) {
        cv::Mat frame;
        auto capture_timestamp = steady_clock::now();
        
        if (!cap.read(frame) || frame.empty()) {
            cerr << "[CAM " << cam_id << "] Failed to read frame or stream ended" << endl;
            break;
        }
        
        // Create timestamped frame
        TimestampedFrame ts_frame;
        ts_frame.frame = frame.clone();
        ts_frame.timestamp = capture_timestamp;
        ts_frame.timestamp_ms = duration_cast<milliseconds>(capture_timestamp - start_time).count();
        ts_frame.frame_index = frame_count;
        ts_frame.camera_id = cam_id;
        
        frame_buffer.Push(ts_frame);
        frame_count++;
        
        // Progress update every 5 seconds
        if (frame_count % (static_cast<int>(fps) * 5) == 0) {
            auto elapsed = duration_cast<seconds>(steady_clock::now() - start_time).count();
            cout << "[CAM " << cam_id << "] Captured " << frame_count << " frames (" << elapsed << "s)" << endl;
        }
    }
    
    cap.release();
    cout << "[CAM " << cam_id << "] Stopped capturing. Total frames: " << frame_count << endl;
}

// Synchronized frame writer
class SynchronizedFrameWriter {
private:
    vector<FrameBuffer*> buffers;
    vector<string> camera_names;
    string output_base_path;
    int64_t sync_tolerance_ms;
    
public:
    SynchronizedFrameWriter(vector<FrameBuffer*> frame_buffers, 
                           vector<string> cam_names, 
                           const string& base_path, 
                           int64_t tolerance_ms = 50) 
        : buffers(frame_buffers), camera_names(cam_names), 
          output_base_path(base_path), sync_tolerance_ms(tolerance_ms) {}
    
    void WriteSynchronizedFrames() {
        cout << "[SYNC] Starting synchronized frame writing with " << sync_tolerance_ms << "ms tolerance" << endl;
        
        // Create output directories
        for (const auto& name : camera_names) {
            fs::create_directories(output_base_path + "/" + name);
        }
        
        int synchronized_frame_count = 0;
        
        while (!stop_requested) {
            // Wait for all buffers to have frames
            bool all_ready = true;
            for (auto buffer : buffers) {
                if (buffer->Size() == 0) {
                    all_ready = false;
                    break;
                }
            }
            
            if (!all_ready) {
                this_thread::sleep_for(milliseconds(10));
                continue;
            }
            
            // Find the latest common timestamp across all cameras
            int64_t reference_timestamp = 0;
            bool first = true;
            
            for (auto buffer : buffers) {
                TimestampedFrame temp_frame;
                if (buffer->GetClosestFrame(0, temp_frame, LLONG_MAX)) {
                    if (first) {
                        reference_timestamp = temp_frame.timestamp_ms;
                        first = false;
                    } else {
                        reference_timestamp = max(reference_timestamp, temp_frame.timestamp_ms);
                    }
                }
            }
            
            if (first) continue; // No frames available
            
            // Extract synchronized frames
            vector<TimestampedFrame> sync_frames(buffers.size());
            bool sync_successful = true;
            
            for (size_t i = 0; i < buffers.size(); i++) {
                if (!buffers[i]->GetClosestFrame(reference_timestamp, sync_frames[i], sync_tolerance_ms)) {
                    sync_successful = false;
                    break;
                }
            }
            
            if (!sync_successful) {
                this_thread::sleep_for(milliseconds(10));
                continue;
            }
            
            // Save synchronized frames
            for (size_t i = 0; i < sync_frames.size(); i++) {
                stringstream ss;
                ss << output_base_path << "/" << camera_names[i] << "/" 
                   << camera_names[i] << "_" << setfill('0') << setw(6) << synchronized_frame_count << ".png";
                
                if (!cv::imwrite(ss.str(), sync_frames[i].frame)) {
                    cerr << "[SYNC] Failed to save frame: " << ss.str() << endl;
                }
            }
            
            synchronized_frame_count++;
            
            // Progress update
            if (synchronized_frame_count % 150 == 0) {
                cout << "[SYNC] Wrote " << synchronized_frame_count << " synchronized frame sets" << endl;
            }
            
            // Optional: Save timestamp info for debugging
            if (synchronized_frame_count % 300 == 0) {
                cout << "[SYNC] Timestamp differences at frame " << synchronized_frame_count << ":" << endl;
                for (size_t i = 0; i < sync_frames.size(); i++) {
                    cout << "  [CAM " << i << "] diff: " << (sync_frames[i].timestamp_ms - reference_timestamp) << "ms" << endl;
                }
            }
        }
        
        cout << "[SYNC] Finished writing " << synchronized_frame_count << " synchronized frame sets" << endl;
    }
};

// Alternative approach: Hardware sync using external trigger
class HardwareSyncCapture {
private:
    vector<cv::VideoCapture> cameras;
    vector<string> rtsp_urls;
    
public:
    HardwareSyncCapture(const vector<string>& urls) : rtsp_urls(urls) {
        cameras.resize(urls.size());
    }
    
    bool Initialize() {
        for (size_t i = 0; i < rtsp_urls.size(); i++) {
            cameras[i].open(rtsp_urls[i]);
            if (!cameras[i].isOpened()) {
                cerr << "[HWSYNC] Failed to open camera " << i << ": " << rtsp_urls[i] << endl;
                return false;
            }
            
            // Set properties for synchronization
            cameras[i].set(cv::CAP_PROP_BUFFERSIZE, 1);
            cameras[i].set(cv::CAP_PROP_FPS, 15); // Force same FPS
            
            cout << "[HWSYNC] Camera " << i << " initialized" << endl;
        }
        return true;
    }
    
    bool CaptureFrameSet(vector<cv::Mat>& frames) {
        frames.resize(cameras.size());
        
        // Capture from all cameras as quickly as possible
        for (size_t i = 0; i < cameras.size(); i++) {
            if (!cameras[i].read(frames[i]) || frames[i].empty()) {
                return false;
            }
        }
        return true;
    }
    
    void Release() {
        for (auto& cam : cameras) {
            cam.release();
        }
    }
};

// Network Time Protocol (NTP) sync approach
class NTPSyncCapture {
private:
    struct CameraStream {
        cv::VideoCapture cap;
        string url;
        steady_clock::time_point start_time;
        double expected_fps;
        int frame_count;
        
        CameraStream(const string& rtsp_url) : url(rtsp_url), frame_count(0) {
            cap.open(url);
            expected_fps = cap.get(cv::CAP_PROP_FPS);
            if (expected_fps <= 0) expected_fps = 15.0;
            start_time = steady_clock::now();
        }
        
        bool GetFrameAtTime(steady_clock::time_point target_time, cv::Mat& frame) {
            auto time_diff = target_time - start_time;
            int target_frame = static_cast<int>(duration_cast<milliseconds>(time_diff).count() * expected_fps / 1000.0);
            
            // Skip frames to reach target time
            while (frame_count < target_frame) {
                cv::Mat temp_frame;
                if (!cap.read(temp_frame)) return false;
                frame_count++;
            }
            
            // Capture the frame at target time
            return cap.read(frame);
        }
    };
    
    vector<unique_ptr<CameraStream>> streams;
    
public:
    NTPSyncCapture(const vector<string>& rtsp_urls) {
        for (const auto& url : rtsp_urls) {
            streams.push_back(make_unique<CameraStream>(url));
        }
    }
    
    bool CaptureAtTime(steady_clock::time_point target_time, vector<cv::Mat>& frames) {
        frames.resize(streams.size());
        
        for (size_t i = 0; i < streams.size(); i++) {
            if (!streams[i]->GetFrameAtTime(target_time, frames[i])) {
                return false;
            }
        }
        return true;
    }
};

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <rtsp_left> <rtsp_center> <rtsp_right> [sync_method]" << endl;
        cerr << "sync_method: 0=timestamp (default), 1=hardware, 2=ntp" << endl;
        return -1;
    }

    signal(SIGINT, signal_handler);
    
    vector<string> rtsp_links = { argv[1], argv[2], argv[3] };
    int sync_method = (argc > 4) ? atoi(argv[4]) : 0;
    
    // Create directory structure
    fs::create_directories("output/frames/left");
    fs::create_directories("output/frames/central");
    fs::create_directories("output/frames/right");
    fs::create_directories("debug");
    
    vector<string> camera_names = {"left", "central", "right"};
    
    cout << "\n=== SYNCHRONIZED RTSP CAPTURE ===" << endl;
    cout << "Sync method: " << sync_method << endl;
    
    if (sync_method == 0) {
        // Timestamp-based synchronization
        vector<FrameBuffer> frame_buffers(3);
        vector<FrameBuffer*> buffer_ptrs = {&frame_buffers[0], &frame_buffers[1], &frame_buffers[2]};
        
        // Start capture threads
        vector<thread> capture_threads;
        for (int i = 0; i < 3; i++) {
            capture_threads.emplace_back(CaptureRTSPWithTimestamps, rtsp_links[i], i, ref(frame_buffers[i]));
        }
        
        // Start synchronized writer
        SynchronizedFrameWriter writer(buffer_ptrs, camera_names, "output/frames", 50); // 50ms tolerance
        thread writer_thread(&SynchronizedFrameWriter::WriteSynchronizedFrames, &writer);
        
        // Wait for completion
        for (auto& t : capture_threads) {
            t.join();
        }
        writer_thread.join();
        
    } else if (sync_method == 1) {
        // Hardware synchronization
        HardwareSyncCapture hw_sync(rtsp_links);
        if (!hw_sync.Initialize()) {
            cerr << "[ERROR] Failed to initialize hardware sync" << endl;
            return -1;
        }
        
        int frame_count = 0;
        while (!stop_requested) {
            vector<cv::Mat> frames;
            if (hw_sync.CaptureFrameSet(frames)) {
                // Save synchronized frames
                for (size_t i = 0; i < frames.size(); i++) {
                    stringstream ss;
                    ss << "output/frames/" << camera_names[i] << "/" << camera_names[i] 
                       << "_" << setfill('0') << setw(6) << frame_count << ".png";
                    cv::imwrite(ss.str(), frames[i]);
                }
                frame_count++;
                
                if (frame_count % 150 == 0) {
                    cout << "[HWSYNC] Captured " << frame_count << " synchronized frame sets" << endl;
                }
            }
        }
        hw_sync.Release();
    }
    
    cout << "\n=== PROCESSING PANORAMA ===" << endl;
    
    // Continue with existing panorama processing...
    StitchingPipeline pipeline;
    if (!pipeline.LoadIntrinsicsData("intrinsic.json") ||
        !pipeline.LoadExtrinsicsData("extrinsic.json")) {
        cerr << "[ERROR] Failed to load calibration data." << endl;
        return -1;
    }
    
    // Process synchronized frames to panorama
    if (!ProcessFramesToPanorama("output/frames", "output/panorama_result.mp4", pipeline, 15.0)) {
        cerr << "[ERROR] Failed to create panorama video" << endl;
        return -1;
    }
    
    cout << "\n=== PROCESSING COMPLETE ===" << endl;
    cout << "Synchronized frames: output/frames/" << endl;
    cout << "Final panorama: output/panorama_result.mp4" << endl;
    
    return 0;
}