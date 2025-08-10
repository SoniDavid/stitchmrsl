# RTSP Multi-Camera Stitching System - Executables Guide

**Note:** This guide assumes the code is run from source directory of the project
## 📖 Overview

This guide covers the two main executables in the RTSP stitching system:

1. **`rtsp_stitcher`** - Multi-camera real-time RTSP stream capturing and stitching
2. **`video_rectifier`** - Single camera video lens distortion correction

Both executables utilize CUDA-accelerated processing for optimal performance and support fisheye camera calibration.

---

## 🎯 **1. RTSP Stitcher (`rtsp_stitcher`)**

### **Purpose**
The `rtsp_stitcher` is a two-phase synchronized processing system that captures frames from 3 RTSP camera streams and creates panoramic videos using CUDA-accelerated stitching.

### **How It Works**

#### **Phase 1: Synchronized Capture**
- **Multi-threaded RTSP capture**: Each camera runs in its own thread for parallel processing
- **Hardware timestamping**: System timestamps are recorded at frame arrival for precise synchronization
- **Asynchronous frame writing**: Captured frames are queued and written to disk by dedicated writer threads
- **Metadata generation**: Each frame gets detailed metadata (timestamp, sequence number, file path)
- **Real-time statistics**: Live FPS monitoring and dropped frame tracking

#### **Phase 2: CUDA-Accelerated Processing**
- **Frame synchronization**: Intelligent algorithm finds optimal frame alignment using timestamps
- **GPU-based rectification**: Fisheye lens distortion correction using CUDA kernels
- **Perspective transformation**: Images are warped to a common coordinate system
- **Multi-mode blending**: Advanced blending techniques (max, average, feathering) for seamless panoramas

### **Usage Modes**

#### **Mode 1: Capture Only**
Captures synchronized frames from 3 RTSP streams and saves them with metadata.

```bash
./build/rtsp_stitcher capture <rtsp_url_1> <rtsp_url_2> <rtsp_url_3> [output_directory]
```

**Arguments:**
- `rtsp_url_1`: RTSP stream URL for camera 1 (MRSL left camera)
- `rtsp_url_2`: RTSP stream URL for camera 2 (MRSL center camera) 
- `rtsp_url_3`: RTSP stream URL for camera 3 (MRSL right camera)
- `output_directory`: (Optional) Directory to save captured frames. Default: `./output`

**Examples:**
```bash
# Basic capture with default output directory
./build/rtsp_stitcher capture rtsp://192.168.1.10/stream rtsp://192.168.1.11/stream rtsp://192.168.1.12/stream

# Capture with custom output directory
./build/rtsp_stitcher capture rtsp://cam1.local/live rtsp://cam2.local/live rtsp://cam3.local/live ./my_capture_session

# Using different RTSP formats
./build/rtsp_stitcher capture rtsp://admin:password@192.168.1.10:554/h264 rtsp://admin:password@192.168.1.11:554/h264 rtsp://admin:password@192.168.1.12:554/h264 ./secure_capture
```

**Terminal Commands:**
```bash
# Terminal 1: Start capture
cd /path/to/stitchmrsl
./build/rtsp_stitcher capture rtsp://camera1 rtsp://camera2 rtsp://camera3

# Terminal 2: Monitor capture progress (in another terminal)
watch -n 1 'ls -la output/raw_frames/*/  | grep -c ".png"'

# Stop capture: Press Ctrl+C in Terminal 1
```

**Output Structure:**
```
output/
├── raw_frames/
│   ├── cam1/
│   │   ├── 00000001.png
│   │   ├── 00000002.png
│   │   └── ...
│   ├── cam2/
│   │   └── ...
│   └── cam3/
│       └── ...
└── metadata/
    ├── cam1_metadata.json
    ├── cam2_metadata.json
    ├── cam3_metadata.json
    └── capture_summary.json
```

#### **Mode 2: Process Only**
Processes previously captured frames into a panoramic video.

```bash
./build/rtsp_stitcher process <metadata_directory> [output_video_path]
```

**Arguments:**
- `metadata_directory`: Directory containing captured frames and metadata
- `output_video_path`: (Optional) Path for output panorama video. Default: `./panorama_result.mp4`

**Examples:**
```bash
# Basic processing with default output
./build/rtsp_stitcher process ./output

# Custom output video name
./build/rtsp_stitcher process ./my_capture_session ./final_panorama.mp4

# Processing with full paths
./build/rtsp_stitcher process /home/user/captures/session_001 /home/user/videos/panorama_20250807.mp4
```

**Terminal Commands:**
```bash
# Terminal 1: Start processing
./build/rtsp_stitcher process ./output ./panorama_result.mp4

# Terminal 2: Monitor GPU usage (optional)
watch -n 1 nvidia-smi

# Terminal 3: Monitor output file size growth
watch -n 5 'ls -lh panorama_result.mp4'
```

#### **Mode 3: Full Pipeline**
Runs both capture and processing phases sequentially.

```bash
./build/rtsp_stitcher full <rtsp_url_1> <rtsp_url_2> <rtsp_url_3> [output_directory] [output_video_path]
```

**Arguments:**
- `rtsp_url_1, rtsp_url_2, rtsp_url_3`: RTSP stream URLs
- `output_directory`: (Optional) Directory for intermediate files. Default: `./output`
- `output_video_path`: (Optional) Final panorama video path. Default: `./panorama_result.mp4`

**Examples:**
```bash
# Complete pipeline with defaults
./build/rtsp_stitcher full rtsp://192.168.1.10/stream rtsp://192.168.1.11/stream rtsp://192.168.1.12/stream

# Complete pipeline with custom paths
./build/rtsp_stitcher full rtsp://cam1/live rtsp://cam2/live rtsp://cam3/live ./capture_data ./final_output.mp4

# Production environment example
./build/rtsp_stitcher full \
  rtsp://admin:pass123@10.0.1.100:554/live \
  rtsp://admin:pass123@10.0.1.101:554/live \
  rtsp://admin:pass123@10.0.1.102:554/live \
  ./production_capture \
  ./production_panorama_$(date +%Y%m%d_%H%M%S).mp4
```

**Terminal Commands:**
```bash
# Terminal 1: Run full pipeline
./build/rtsp_stitcher full rtsp://cam1 rtsp://cam2 rtsp://cam3

# Terminal 2: Monitor system resources
htop

# Terminal 3: Monitor disk space
watch -n 10 'df -h .'

# To stop gracefully: Press Ctrl+C in Terminal 1 during capture phase
```

### **Required Files**
- **`intrinsic.json`**: Camera calibration parameters (must be in current directory)
- **`extrinsic.json`**: Camera positioning and transformation matrices (must be in current directory)

### **Calibration File Format**

**intrinsic.json** example:
```json
{
  "cameras": {
    "izquierda": {
      "K": [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
      "dist": [[-0.1], [0.05], [-0.02], [0.01]],
      "image_size": [640, 480],
      "model": "fisheye"
    },
    "central": {
      "K": [[405.0, 0.0, 320.0], [0.0, 405.0, 240.0], [0.0, 0.0, 1.0]],
      "dist": [[-0.12], [0.06], [-0.015], [0.008]],
      "image_size": [640, 480],
      "model": "fisheye"
    },
    "derecha": {
      "K": [[398.0, 0.0, 320.0], [0.0, 398.0, 240.0], [0.0, 0.0, 1.0]],
      "dist": [[-0.11], [0.055], [-0.018], [0.009]],
      "image_size": [640, 480],
      "model": "fisheye"
    }
  }
}
```

**extrinsic.json** example:
```json
{
  "best_transforms": {
    "AB_similarity": [
      [0.98, -0.02, 50.0],
      [0.02, 0.98, 5.0],
      [0.0, 0.0, 1.0]
    ],
    "BC_similarity": [
      [0.99, 0.01, -45.0],
      [-0.01, 0.99, -2.0],
      [0.0, 0.0, 1.0]
    ]
  }
}
```

---

## 🎯 **2. Video Rectifier (`video_rectifier`)**

### **Purpose**
The `video_rectifier` corrects lens distortion in pre-recorded videos from fisheye cameras using CUDA-accelerated processing.

### **How It Works**

#### **Initialization Phase**
- **CUDA device detection**: Verifies GPU availability and capabilities
- **Calibration loading**: Reads camera-specific parameters from `intrinsic.json`
- **Rectification map generation**: Pre-computes undistortion maps using fisheye model
- **GPU memory allocation**: Prepares CUDA buffers for frame processing

#### **Processing Phase**
- **Frame-by-frame processing**: Reads input video sequentially
- **GPU upload**: Transfers frames to GPU memory
- **CUDA rectification**: Applies lens correction using pre-computed maps
- **GPU download**: Retrieves corrected frames back to CPU
- **Video encoding**: Writes corrected frames to output video

### **Usage**

```bash
./build/video_rectifier <input_video> <camera_name> [output_video]
```

**Arguments:**
- `input_video`: Path to the input video file with lens distortion
- `camera_name`: Camera identifier that matches calibration data (`izquierda`, `central`, `derecha`)
- `output_video`: (Optional) Path for corrected output video. Default: auto-generated

**Examples:**
```bash
# Basic rectification with auto-generated output name
./build/video_rectifier raw_footage.mp4 central
# Output: raw_footage_central_rectified.mp4

# Custom output filename
./build/video_rectifier fisheye_video.mp4 izquierda corrected_left_camera.mp4

# Process multiple cameras sequentially
./build/video_rectifier left_cam.mp4 izquierda left_corrected.mp4
./build/video_rectifier center_cam.mp4 central center_corrected.mp4
./build/video_rectifier right_cam.mp4 derecha right_corrected.mp4

# Full path examples
./build/video_rectifier /home/user/videos/input.mp4 central /home/user/output/rectified.mp4
```

### **Terminal Commands**

#### **Single Camera Processing**
```bash
# Terminal 1: Start rectification
./build/video_rectifier camera1_recording.mp4 izquierda

# Terminal 2: Monitor GPU usage
watch -n 1 nvidia-smi

# Terminal 3: Monitor progress via file size
watch -n 5 'ls -lh *_rectified.mp4'
```

#### **Batch Processing Multiple Videos**
```bash
# Terminal 1: Create batch processing script
cat > process_all_cameras.sh << 'EOF'
#!/bin/bash
echo "Processing left camera..."
./video_rectifier left_camera.mp4 izquierda left_rectified.mp4

echo "Processing center camera..."
./video_rectifier center_camera.mp4 central center_rectified.mp4

echo "Processing right camera..."
./video_rectifier right_camera.mp4 derecha right_rectified.mp4

echo "All cameras processed!"
EOF

chmod +x process_all_cameras.sh

# Terminal 1: Run batch processing
./process_all_cameras.sh

# Terminal 2: Monitor system resources
htop
```

#### **Parallel Processing (Advanced)**
```bash
# Terminal 1: Process left camera
./video_rectifier left_cam.mp4 izquierda left_rect.mp4 &
LEFT_PID=$!

# Terminal 2: Process center camera  
./video_rectifier center_cam.mp4 central center_rect.mp4 &
CENTER_PID=$!

# Terminal 3: Process right camera
./video_rectifier right_cam.mp4 derecha right_rect.mp4 &
RIGHT_PID=$!

# Terminal 4: Monitor all processes
wait $LEFT_PID && echo "Left camera done"
wait $CENTER_PID && echo "Center camera done"  
wait $RIGHT_PID && echo "Right camera done"
echo "All rectification complete!"
```

### **Required Files**
- **`intrinsic.json`**: Must be in the current directory with calibration for specified camera

### **Output Naming Convention**
If no output filename is specified, the tool automatically generates:
```
<input_stem>_<camera_name>_rectified<extension>
```

Examples:
- `input.mp4` + `central` → `input_central_rectified.mp4`
- `fisheye_recording.avi` + `izquierda` → `fisheye_recording_izquierda_rectified.avi`

---

## 🔧 **System Requirements**

### **Hardware Requirements**
- **NVIDIA GPU**: CUDA Compute Capability 6.1 or higher
- **GPU Memory**: Minimum 2GB for 720p, 4GB recommended for 1080p processing
- **System RAM**: Minimum 8GB, 16GB recommended for multi-camera processing
- **Storage**: SSD recommended for real-time capture (HDD may cause frame drops)
- **Network**: Gigabit Ethernet for multiple RTSP streams

### **Software Requirements**
- **CUDA Toolkit**: Version 11.0 or later
- **OpenCV**: Version 4.5+ with CUDA support enabled
- **CMake**: Version 3.12+
- **GCC**: Version 7+ with C++17 support
- **nlohmann/json**: For JSON configuration parsing

### **RTSP Stream Requirements**
- **Supported Codecs**: H.264, H.265 (depends on OpenCV build)
- **Resolution**: Up to 4K (limited by GPU memory)
- **Frame Rate**: Up to 60 FPS (limited by processing capability)
- **Network Latency**: < 100ms for optimal synchronization

---

## 🚀 **Build Instructions**

```bash
# Clone and navigate to project
git clone <repository_url>
cd stitchmrsl

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build executables
make -j$(nproc)

# Executables will be in build directory:
# - rtsp_stitcher
# - video_rectifier
```

---

## 📊 **Performance Tips**

### **For RTSP Stitcher**
1. **Network Optimization**:
   - Use wired connections for cameras
   - Configure cameras for consistent frame rates
   - Monitor network bandwidth usage

2. **Storage Optimization**:
   - Use SSD for capture directory
   - Monitor available disk space during long captures
   - Consider using separate drives for each camera

3. **GPU Optimization**:
   - Ensure sufficient GPU memory
   - Monitor GPU utilization with `nvidia-smi`
   - Close other GPU-intensive applications

### **For Video Rectifier**
1. **Processing Speed**:
   - Process multiple videos in parallel if GPU memory allows
   - Use NVMe SSD for input/output files
   - Process lower resolution videos first for testing

2. **Quality Settings**:
   - Verify calibration parameters before processing
   - Test with short clips before processing long videos
   - Monitor output for quality issues

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **CUDA Not Found**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check OpenCV CUDA support
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i cuda
```

#### **RTSP Connection Issues**
```bash
# Test RTSP stream with ffmpeg
ffprobe rtsp://your_camera_url

# Test with OpenCV directly
python3 -c "import cv2; cap = cv2.VideoCapture('rtsp://your_url'); print('Connected:', cap.isOpened())"
```

#### **Calibration File Issues**
- Ensure `intrinsic.json` and `extrinsic.json` are in the working directory
- Validate JSON syntax with online JSON validators
- Check camera names match exactly (case-sensitive)

#### **Memory Issues**
```bash
# Monitor memory usage
watch -n 1 'free -h'

# Monitor GPU memory
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

### **Performance Monitoring**

#### **Real-time Monitoring Script**
```bash
# Create monitoring script
cat > monitor_stitching.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    echo "Disk Usage:"
    df -h . | tail -1
    echo "Network:"
    ss -i | grep rtsp || echo "No RTSP connections"
    echo "Captured Frames:"
    find output/raw_frames -name "*.png" 2>/dev/null | wc -l
    echo "========================"
    sleep 5
done
EOF

chmod +x monitor_stitching.sh
./monitor_stitching.sh
```

---

## 📚 **Example Workflows**

### **Complete Production Workflow**

```bash
# 1. Setup environment
cd /path/to/stitchmrsl
mkdir -p captures/$(date +%Y%m%d)
cd captures/$(date +%Y%m%d)

# 2. Copy calibration files
cp ../../intrinsic.json .
cp ../../extrinsic.json .

# 3. Start monitoring in background
../../monitor_stitching.sh > monitoring.log &
MONITOR_PID=$!

# 4. Run capture (replace URLs with actual camera URLs)
../../rtsp_stitcher capture \
  rtsp://admin:password@192.168.1.100/live \
  rtsp://admin:password@192.168.1.101/live \
  rtsp://admin:password@192.168.1.102/live \
  ./capture_data

# 5. Process captured data
../../rtsp_stitcher process ./capture_data ./final_panorama.mp4

# 6. Stop monitoring
kill $MONITOR_PID

# 7. Generate summary report
echo "Capture completed at $(date)" > session_report.txt
echo "Frames captured:" >> session_report.txt
find capture_data/raw_frames -name "*.png" | wc -l >> session_report.txt
echo "Final video size:" >> session_report.txt
ls -lh final_panorama.mp4 >> session_report.txt
```

### **Video Rectification Workflow**

```bash
# 1. Prepare input videos
mkdir -p raw_videos rectified_videos

# 2. Verify calibration
if [ ! -f intrinsic.json ]; then
    echo "Error: intrinsic.json not found"
    exit 1
fi

# 3. Process all camera videos
for camera in izquierda central derecha; do
    input_video="raw_videos/${camera}_recording.mp4"
    output_video="rectified_videos/${camera}_rectified.mp4"
    
    if [ -f "$input_video" ]; then
        echo "Processing $camera camera..."
        ./video_rectifier "$input_video" "$camera" "$output_video"
        echo "Completed $camera camera"
    else
        echo "Warning: $input_video not found, skipping"
    fi
done

echo "All rectification complete!"
ls -lh rectified_videos/
```

This comprehensive guide provides detailed information on both executables, covering their purposes, technical implementation, usage patterns, and practical examples for various scenarios.
