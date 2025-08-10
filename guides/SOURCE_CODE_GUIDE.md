# Source Code Architecture Guide - `src/` Directory

## 📖 **Overview**

This guide provides detailed explanations of how each source file in the `src/` directory works. The documentation is designed for developers who want to understand the internal architecture and implementation details of the RTSP multi-camera stitching system.

---

## 🎯 **Main Executables**

### **`main.cc` - GUI Application Entry Point**

**Purpose**: Initializes and runs the interactive Dear ImGui application for real-time camera calibration and stitching.

#### **How It Works**:

1. **Graphics Context Setup**:
   ```cpp
   // Cross-platform OpenGL version detection
   #if defined(__APPLE__)
       const char* glsl_version = "#version 150";  // macOS compatibility
   #else
       const char* glsl_version = "#version 330 core";  // Linux/Windows
   #endif
   ```

2. **Window Creation and DPI Handling**:
   ```cpp
   // Automatically detects monitor DPI and scales interface
   float xscale, yscale;
   glfwGetMonitorContentScale(primary_monitor, &xscale, &yscale);
   int window_width = (int)(1600 * main_scale);
   ```

3. **OpenGL Context Management**:
   - Initializes GLFW for window management
   - Sets up OpenGL 3.3+ context with proper versioning
   - Configures GL3W for OpenGL extension loading

4. **GUI Integration**:
   - Initializes Dear ImGui rendering backend
   - Sets up ImGui for GLFW and OpenGL3
   - Handles DPI scaling for high-resolution displays

#### **Key Features**:
- **Cross-platform compatibility** (Windows, Linux, macOS)
- **Automatic DPI scaling** for different monitor types
- **OpenGL error handling** with proper fallbacks
- **Window lifecycle management** with graceful cleanup

---

### **`rtsp_stitcher.cc` - Command-Line RTSP Processing Tool**

**Purpose**: Production-ready executable for capturing RTSP streams and creating panoramic videos.

#### **Architecture Overview**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mode Parser   │───▶│  Capture Phase  │───▶│ Processing Phase│
│ (capture/process│    │ (Multi-threaded)│    │ (CUDA Pipeline) │
│    /full)       │    └─────────────────┘    └─────────────────┘
└─────────────────┘            │                       │
                               ▼                       ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │ Frame Storage + │    │ Panorama Video  │
                    │ Metadata        │    │ Output          │
                    └─────────────────┘    └─────────────────┘
```

#### **Three Operational Modes**:

1. **Capture Mode** (`mode_capture()`):
   ```cpp
   std::vector<std::string> rtsp_urls = {argv[2], argv[3], argv[4]};
   g_capture_system = std::make_unique<SynchronizedCapture>();
   g_capture_system->CaptureAllStreams(rtsp_urls, output_dir);
   ```
   - **Multi-threaded RTSP capture** from 3 cameras
   - **Hardware timestamping** at frame arrival
   - **Asynchronous frame writing** to prevent blocking
   - **Metadata generation** for synchronization

2. **Process Mode** (`mode_process()`):
   ```cpp
   FrameSynchronizer synchronizer;
   auto synchronized_frames = synchronizer.FindOptimalSync(metadata);
   create_panorama_video(synchronized_frames, output_video);
   ```
   - **Loads captured frame metadata** from JSON files
   - **Intelligent frame synchronization** using timestamps
   - **CUDA-accelerated stitching** pipeline
   - **Multi-format video output** (MP4, AVI, etc.)

3. **Full Mode** (`mode_full()`):
   - **Sequential execution** of capture then process
   - **Automatic data handoff** between phases
   - **End-to-end automation** for production use

#### **Signal Handling**:
```cpp
void signal_handler(int signal) {
    if (signal == SIGINT) {
        g_stop_requested = true;
        if (g_capture_system) {
            g_capture_system->StopCapture();
        }
    }
}
```
- **Graceful shutdown** on Ctrl+C
- **Metadata preservation** during interruption
- **Resource cleanup** and statistics reporting

---

### **`video_rectifier.cc` - Single Camera Distortion Correction**

**Purpose**: CUDA-accelerated fisheye lens distortion correction for pre-recorded videos.

#### **Processing Pipeline**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Video Input     │───▶│ CUDA Rectifier  │───▶│ Corrected Video │
│ (Fisheye)       │    │ Pipeline        │    │ Output          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Frame-by-Frame  │    │ GPU Memory      │    │ Video Writer    │
│ Reading         │    │ Management      │    │ (Same FPS/Format│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### **Key Components**:

1. **VideoRectifier Class**:
   ```cpp
   class VideoRectifier {
       cv::cuda::GpuMat gpu_input_frame_;      // GPU input buffer
       cv::cuda::GpuMat gpu_rectified_frame_;  // GPU output buffer
       cv::cuda::GpuMat gpu_map1_, gpu_map2_;  // Rectification maps
   };
   ```

2. **Calibration Loading**:
   ```cpp
   bool LoadCalibration() {
       // Loads camera-specific parameters from intrinsic.json
       cv::fisheye::estimateNewCameraMatrixForUndistortRectify();
       cv::fisheye::initUndistortRectifyMap();  // Pre-compute maps
   }
   ```

3. **CUDA Processing Loop**:
   ```cpp
   while (cap.read(cpu_frame)) {
       gpu_input_frame_.upload(cpu_frame);        // CPU → GPU
       cv::cuda::remap(gpu_input_frame_, gpu_rectified_frame_, 
                      gpu_map1_, gpu_map2_);      // GPU rectification
       gpu_rectified_frame_.download(cpu_rectified); // GPU → CPU
       writer.write(cpu_rectified);               // Write to video
   }
   ```

#### **Performance Features**:
- **Pre-computed rectification maps** for maximum speed
- **GPU memory reuse** between frames
- **Real-time progress reporting** with FPS statistics
- **Automatic output path generation** based on camera name

---

## 🔧 **Core System Components**

### **`synchronized_capture.cc` - Multi-Camera RTSP Capture System**

**Purpose**: Handles synchronized capture from multiple RTSP streams with precise timestamping and asynchronous I/O.

#### **Architecture Design**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera 1      │    │   Camera 2      │    │   Camera 3      │
│  Capture Thread │    │  Capture Thread │    │  Capture Thread │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│               Shared Write Queue                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Frame 1   │  │   Frame 2   │  │   Frame N   │   ...       │
│  │ + Metadata  │  │ + Metadata  │  │ + Metadata  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐    ┌─────────────────┐
│  Writer Thread  │    │  Writer Thread  │    (Multiple for speed)
│       1         │    │       N         │
└─────────────────┘    └─────────────────┘
```

#### **Key Implementation Details**:

1. **Thread-Safe Capture**:
   ```cpp
   void CaptureCamera(int camera_id, const std::string& rtsp_url) {
       cv::VideoCapture cap(rtsp_url);
       cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // Minimize buffering
       
       while (!stop_requested_) {
           auto capture_time = std::chrono::high_resolution_clock::now();
           if (cap.read(frame)) {
               // Queue frame with precise timestamp
               QueueFrameForWriting(frame, capture_time, camera_id);
           }
       }
   }
   ```

2. **Asynchronous I/O Management**:
   ```cpp
   void AsyncFrameWriter() {
       while (!stop_requested_ || !write_queue_.empty()) {
           WriteTask task = GetNextWriteTask();
           cv::imwrite(task.filename, task.frame, compression_params_);
       }
   }
   ```

3. **Metadata Generation**:
   ```cpp
   struct FrameMetadata {
       std::chrono::high_resolution_clock::time_point timestamp;
       uint64_t sequence_number;
       std::string frame_path;
       int camera_id;
       cv::Size frame_size;
       bool is_valid;
   };
   ```

#### **Performance Optimizations**:
- **Configurable buffer sizes** to prevent memory overflow
- **Multi-threaded frame writing** using hardware thread count
- **PNG compression optimization** for storage efficiency
- **Real-time statistics tracking** (FPS, dropped frames)

---

### **`frame_synchronizer.cc` - Intelligent Frame Alignment**

**Purpose**: Analyzes captured frame metadata to find optimal synchronized frame triplets for stitching.

#### **Synchronization Algorithm**:

```
Input: 3 streams of timestamped frames
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Camera 1    │    │     Camera 2    │    │     Camera 3    │
│ T1: 10.000ms    │    │ T1: 10.005ms    │    │ T1: 10.002ms    │
│ T2: 10.033ms    │    │ T2: 10.040ms    │    │ T2: 10.035ms    │
│ T3: 10.067ms    │    │ T3: 10.070ms    │    │ T3: 10.068ms    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Synchronization Analysis                           │
│  • Find frames within sync window (±50ms default)              │
│  • Calculate sync quality (timestamp deviation)                │
│  • Filter by minimum quality threshold                         │
│  • Generate optimal triplet sequence                           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
Output: Synchronized frame triplets
┌─────────────────────────────────────────────────────────────────┐
│ Triplet 1: [Frame_1_001, Frame_2_001, Frame_3_001] (±2ms)     │
│ Triplet 2: [Frame_1_002, Frame_2_002, Frame_3_002] (±5ms)     │
│ Triplet N: [Frame_1_N,   Frame_2_N,   Frame_3_N]   (±3ms)     │
└─────────────────────────────────────────────────────────────────┘
```

#### **Implementation Details**:

1. **Time Window Matching**:
   ```cpp
   std::vector<SyncedFrameTriplet> FindOptimalSync(
       const std::vector<FrameMetadata>& cam1_meta,
       const std::vector<FrameMetadata>& cam2_meta, 
       const std::vector<FrameMetadata>& cam3_meta) {
       
       // Sort all frames by timestamp
       SortFramesByTimestamp(cam1_sorted);
       
       // Find overlapping time ranges
       auto earliest_start = std::max({cam1_start, cam2_start, cam3_start});
       auto latest_end = std::min({cam1_end, cam2_end, cam3_end});
   }
   ```

2. **Quality Assessment**:
   ```cpp
   double CalculateSyncQuality(const TimePoint& t1, const TimePoint& t2, const TimePoint& t3) {
       auto max_time = std::max({t1, t2, t3});
       auto min_time = std::min({t1, t2, t3});
       return std::chrono::duration_cast<std::chrono::milliseconds>(max_time - min_time).count();
   }
   ```

3. **Configurable Parameters**:
   ```cpp
   void SetSyncWindow(std::chrono::milliseconds window);      // ±50ms default
   void SetMinSyncQuality(double max_deviation_ms);          // 100ms default
   void SetMaxFrameAge(std::chrono::milliseconds max_age);   // 200ms default
   ```

#### **Quality Metrics**:
- **Perfect sync**: < 5ms deviation between cameras
- **Good sync**: < 16.7ms deviation (one frame at 60fps)
- **Acceptable sync**: < configurable threshold
- **Statistics tracking**: Average quality, effective FPS, triplet counts

---

### **`cuda_stitching_pipeline.cc` - GPU-Accelerated Stitching Engine**

**Purpose**: High-performance panoramic stitching using CUDA-accelerated OpenCV operations.

#### **Processing Pipeline**:

```
Input: 3 synchronized camera frames
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Raw Frame 1    │    │  Raw Frame 2    │    │  Raw Frame 3    │
│  (Fisheye)      │    │  (Fisheye)      │    │  (Fisheye)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ GPU Rectify 1   │    │ GPU Rectify 2   │    │ GPU Rectify 3   │
│ (Undistort)     │    │ (Undistort)     │    │ (Undistort)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  GPU Warp 1     │    │  GPU Warp 2     │    │  GPU Warp 3     │
│ (Transform)     │    │ (Identity)      │    │ (Transform)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────┬───────────┴──────────┬───────────┘
                     │                      │
                     ▼                      ▼
              ┌─────────────────────────────────────┐
              │        GPU Blending Engine          │
              │ • Max blending (fastest)            │
              │ • Average blending (smooth)         │
              │ • Feathering blending (seamless)    │
              └─────────────────┬───────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │    Panorama Output      │
                    │   (Auto-cropped)        │
                    └─────────────────────────┘
```

#### **Key Implementation Features**:

1. **GPU Memory Management**:
   ```cpp
   bool AllocateGPUMemory(const cv::Size& input_size) {
       for (int i = 0; i < 3; ++i) {
           gpu_frames_[i].create(input_size, CV_8UC3);       // Input frames
           gpu_rectified_[i].create(input_size, CV_8UC3);    // Rectified frames
           gpu_warped_[i].create(output_size_, CV_8UC3);     // Warped frames
           gpu_weight_masks_[i].create(output_size_, CV_32F); // Blend weights
       }
   }
   ```

2. **Fisheye Rectification**:
   ```cpp
   bool RectifyImagesGPU() {
       for (int i = 0; i < 3; ++i) {
           cv::cuda::remap(gpu_frames_[i], gpu_rectified_[i], 
                          gpu_map1_[i], gpu_map2_[i], 
                          cv::INTER_LINEAR);
       }
   }
   ```

3. **Perspective Transformation**:
   ```cpp
   cv::Mat StitchImagesGPU() {
       // Calculate global canvas size from corner transformations
       cv::perspectiveTransform(corners, transformed_corners, transform_matrix);
       
       // Warp each camera to common coordinate system
       cv::cuda::warpPerspective(gpu_rectified_[i], gpu_warped_[i], 
                                final_transform, output_size_);
   }
   ```

4. **Advanced Blending Modes**:
   ```cpp
   // Mode 0: Max blending (fastest)
   cv::cuda::max(gpu_warped_[0], gpu_warped_[1], temp_result);
   cv::cuda::max(temp_result, gpu_warped_[2], gpu_panorama_);
   
   // Mode 2: Feathering blending (seamless)
   GenerateWeightMasks();  // Distance-based weights
   ApplyFeatheringBlend(); // Weighted accumulation
   ```

#### **Performance Optimizations**:
- **Pre-computed rectification maps** uploaded to GPU
- **Memory reuse** between processing calls
- **Automatic canvas size calculation** for optimal output
- **GPU-based auto-cropping** to remove black borders

---

### **`async_frame_writer.cc` - High-Performance I/O System**

**Purpose**: Provides non-blocking frame writing capabilities to prevent capture thread stalling.

#### **Architecture**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Capture Thread  │    │ Capture Thread  │    │ Capture Thread  │
│    Camera 1     │    │    Camera 2     │    │    Camera 3     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Thread-Safe Write Queue                     │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│ │ Frame + │ │ Frame + │ │ Frame + │ │ Frame + │ │ Frame + │    │
│ │ Path +  │ │ Path +  │ │ Path +  │ │ Path +  │ │ Path +  │    │
│ │ Meta    │ │ Meta    │ │ Meta    │ │ Meta    │ │ Meta    │    │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘    │
└─────────┬───────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Writer Thread Pool                           │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │Writer Thread│ │Writer Thread│ │Writer Thread│ │Writer Thread│ │
│ │     1       │ │     2       │ │     3       │ │     N       │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Disk Storage  │    │   Disk Storage  │    │   Disk Storage  │
│   Camera 1      │    │   Camera 2      │    │   Camera 3      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### **Implementation Details**:

1. **Thread-Safe Queue Management**:
   ```cpp
   struct WriteTask {
       cv::Mat frame;
       std::string filename;
       std::chrono::high_resolution_clock::time_point timestamp;
       int camera_id;
       uint64_t sequence_number;
   };
   
   std::queue<WriteTask> write_queue_;
   std::mutex queue_mutex_;
   std::condition_variable queue_cv_;
   ```

2. **Configurable Compression**:
   ```cpp
   void SetCompressionParams(const std::vector<int>& params) {
       compression_params_ = params;
       // Example: {cv::IMWRITE_PNG_COMPRESSION, 6}
       //          {cv::IMWRITE_JPEG_QUALITY, 85}
   }
   ```

3. **Performance Statistics**:
   ```cpp
   struct WriterStats {
       size_t frames_written;
       size_t frames_dropped;
       uint64_t total_bytes_written;
       double average_write_time_ms;
       double current_write_rate_fps;
   };
   ```

#### **Features**:
- **Automatic thread count** based on hardware capabilities
- **Queue overflow protection** with configurable limits
- **Compression optimization** for different storage scenarios
- **Real-time performance monitoring** and statistics

---

### **`error_handler.cc` - Centralized Error Management**

**Purpose**: Provides unified error logging, tracking, and recovery mechanisms across all system components.

#### **Error Level Hierarchy**:
```cpp
enum class ErrorLevel {
    DEBUG,    // Development and troubleshooting information
    INFO,     // General operational information
    WARNING,  // Non-critical issues that should be monitored
    ERROR,    // Critical issues that affect functionality
    CRITICAL  // System-threatening errors requiring immediate attention
};
```

#### **Key Features**:

1. **Multi-Destination Logging**:
   ```cpp
   void Log(ErrorLevel level, const std::string& component, 
           const std::string& message, const std::string& file, int line) {
       
       // Console output with color coding
       if (level >= ErrorLevel::ERROR) {
           std::cerr << log_line << std::endl;  // stderr for errors
       } else {
           std::cout << log_line << std::endl;  // stdout for info
       }
       
       // File logging with rotation
       if (!log_file_path_.empty()) {
           std::ofstream log_file(log_file_path_, std::ios::app);
           log_file << log_line << std::endl;
       }
   }
   ```

2. **Contextual Information**:
   ```cpp
   // Automatic file/line information
   #define LOG_ERROR(component, message) \
       ErrorHandler::Log(ErrorLevel::ERROR, component, message, __FILE__, __LINE__)
   ```

3. **Critical Error Tracking**:
   ```cpp
   static bool HasCriticalErrors() { return has_critical_errors_; }
   static void ClearErrorState() { has_critical_errors_ = false; }
   ```

---

### **`stitching.cc` - CPU Fallback Stitching Implementation**

**Purpose**: Provides CPU-based stitching algorithms as fallback when CUDA is unavailable or for debugging.

#### **Key Differences from CUDA Pipeline**:

1. **Memory Management**:
   ```cpp
   // CPU matrices instead of GPU memory
   cv::Mat rectified_frames[3];
   cv::Mat warped_frames[3];
   cv::Mat panorama_result;
   ```

2. **Processing Functions**:
   ```cpp
   // CPU-based operations
   cv::remap(input, rectified, map1, map2, cv::INTER_LINEAR);
   cv::warpPerspective(rectified, warped, transform, output_size);
   ```

3. **Blending Implementation**:
   ```cpp
   // CPU blending algorithms
   cv::max(warped1, warped2, temp);
   cv::max(temp, warped3, final_result);
   ```

#### **Use Cases**:
- **Development and debugging** without GPU requirements
- **Fallback processing** when CUDA drivers fail
- **Algorithm verification** and testing
- **Low-power deployment** scenarios

---

## 🔍 **Integration and Data Flow**

### **Overall System Integration**:

#### **GUI Application Dependencies**:
```
┌─────────────────┐
│     main.cc     │ (GUI Entry Point)
│                 │
├─────────────────┤
│   stitching.cc  │ (CPU Pipeline - Fallback)
│                 │
├─────────────────┤
│cuda_stitching_  │ (GPU Pipeline - Primary)
│  pipeline.cc    │
└─────────────────┘
```

#### **RTSP Stitcher Dependencies**:
```
┌─────────────────────────────────────────────────────────────┐
│                    rtsp_stitcher.cc                        │
│                  (Main Executable)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│synchronized_│ │frame_       │ │cuda_        │
│capture.cc   │ │synchronizer │ │stitching_   │
│             │ │.cc          │ │pipeline.cc  │
└─────┬───────┘ └─────────────┘ └─────┬───────┘
      │                               │
      ▼                               ▼
┌─────────────┐                ┌─────────────┐
│async_frame_ │                │error_       │
│writer.cc    │                │handler.cc   │
└─────┬───────┘                └─────────────┘
      │                              
      ▼                              
┌─────────────┐                      
│error_       │                      
│handler.cc   │                      
└─────────────┘                      
```

#### **Video Rectifier Dependencies**:
```
┌─────────────────┐
│video_rectifier  │ (Single Camera Tool)
│.cc              │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│cuda_stitching_  │ (GPU Processing)
│pipeline.cc      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│error_handler.cc │ (Logging & Errors)
└─────────────────┘
```

### **Data Flow Patterns**:

1. **Capture Flow**: RTSP → synchronized_capture → async_frame_writer → Disk
2. **Sync Flow**: Disk metadata → frame_synchronizer → Triplet list
3. **Processing Flow**: Triplet list → cuda_stitching_pipeline → Panorama video
4. **Error Flow**: All components → error_handler → Console/Log file

### **Thread Safety**:
- **Mutex protection** for all shared data structures
- **Atomic variables** for simple state flags
- **Condition variables** for efficient thread coordination
- **RAII patterns** for automatic resource cleanup

---

## 🏆 **Best Practices Demonstrated**

### **Performance Optimization**:
- **GPU memory reuse** to minimize allocation overhead
- **Pre-computed lookup tables** for real-time processing
- **Multi-threading** for parallel I/O and computation
- **Memory-mapped file access** for large datasets

### **Error Handling**:
- **Graceful degradation** when components fail
- **Resource cleanup** using RAII patterns
- **Comprehensive logging** for debugging and monitoring
- **State validation** at critical checkpoints

### **Code Organization**:
- **Single responsibility principle** for each source file
- **Clean interfaces** between components
- **Consistent naming conventions** throughout
- **Comprehensive documentation** for public APIs

This architecture provides a robust, scalable foundation for real-time multi-camera panoramic stitching with excellent performance characteristics and maintainability.
