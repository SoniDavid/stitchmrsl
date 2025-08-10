# RTSP Multi-Camera Stitching System

## 🎯 **Project Overview**

This is a high-performance, real-time multi-camera panoramic stitching system designed for RTSP surveillance and monitoring applications. The system captures synchronized video streams from three fisheye cameras and creates seamless panoramic videos using CUDA-accelerated processing.

### **📚 Documentation Resources:**
- **📖 [Executables Usage Guide](guides/EXECUTABLES_GUIDE.md)** - Detailed command-line usage and examples
- **🔧 [Source Code Architecture Guide](guides/SOURCE_CODE_GUIDE.md)** - Internal implementation details
- **📁 [Ai generated documentation](docs/)** - Component-specific documentation for a "simplier" approach to understand the project source code
- **🎯 [Sample Images](imgs/)** - Test data and calibration examples

### **Key Features:**
- **Real-time RTSP capture** from 3 cameras with hardware timestamping
- **CUDA-accelerated processing** for fisheye correction and panoramic stitching  
- **Two-phase architecture** separating capture from processing for reliability
- **Advanced frame synchronization** handling network jitter and frame drops
- **Multiple blending modes** (max, average, feathering) for optimal visual quality
- **Calibration-driven workflow** supporting fisheye camera models
- **GUI and command-line interfaces** for different use cases

### **System Architecture:**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera 1      │───▶│ Synchronized    │───▶│ CUDA Stitching  │
│   (RTSP Feed)   │    │ Capture System  │    │ Pipeline        │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│   Camera 2      │───▶│ • Timestamping  │───▶│ • Rectification │
│   (RTSP Feed)   │    │ • Frame Sync    │    │ • Warping       │
├─────────────────┤    │ • Metadata      │    │ • Blending      │
│   Camera 3      │───▶│ • Async I/O     │───▶│ • Auto-cropping │
│   (RTSP Feed)   │    └─────────────────┘    └─────────────────┘
└─────────────────┘                                    │
                                                       ▼
                                           ┌─────────────────┐
                                           │ Panoramic Video │
                                           │ Output          │
                                           └─────────────────┘
```

## 🏗️ **Project Structure**

### **Root Directory**
```
stitchmrsl/
├── README.md                    # This file - main project documentation
├── EXECUTABLES_GUIDE.md         # Detailed usage guide for built executables
├── CMakeLists.txt              # Build configuration and dependencies
├── .gitignore                  # Git ignore patterns
├── .gitmodules                 # Git submodules configuration
├── intrinsic.json              # Camera calibration parameters
├── extrinsic.json              # Camera positioning and transformations
├── stitch.py                   # Python offline stitching and calibration
├── imgui.ini                   # GUI layout and preferences
├── raw_video_gen.sh            # Shell script for test video generation
├── cpp.jpg                     # C++ workflow demonstration image
└── python.jpg                  # Python workflow demonstration image
```

### **Source Code Organization**

#### **`src/` - Implementation Files**
```
src/
├── main.cc                      # Main GUI application entry point
├── app.cc                       # Application state and coordination
├── gui.cc                       # Dear ImGui interface implementation
├── stitching.cc                 # Core CPU stitching algorithms
│
├── rtsp_stitcher.cc            # 🎯 RTSP capture and processing executable
├── video_rectifier.cc          # 🎯 Single camera rectification tool
│
├── synchronized_capture.cc      # Multi-threaded RTSP capture system
├── frame_synchronizer.cc       # Intelligent frame alignment algorithms
├── async_frame_writer.cc       # High-performance frame I/O
├── cuda_stitching_pipeline.cc  # CUDA-accelerated stitching engine
├── error_handler.cc            # Error management and recovery
│
├── cuda_stitching_pipeline_old.cc # Legacy CUDA implementation
└── rtsp_stitcher_old.cc        # Legacy RTSP implementation
```

#### **`include/` - Header Files**
```
include/
├── app.hh                      # Application state management
├── gui.hh                      # GUI interface definitions
├── stitching.hh                # Core stitching algorithms
│
├── synchronized_capture.hh     # RTSP capture system interface
├── frame_synchronizer.hh       # Frame synchronization algorithms
├── async_frame_writer.hh       # Asynchronous I/O operations
├── cuda_stitching.hh          # CUDA processing pipeline
└── error_handler.hh           # Error handling utilities
```

### **Documentation**

#### **`docs/` - Technical Documentation**
```
docs/
├── index.md                                                    # Documentation index
├── 01_offline_stitching___calibration_script__python__.md    # Python calibration guide
├── 02_applicationstate__gui__.md                             # GUI application documentation
├── 03_stitchingpipeline__c____.md                           # C++ stitching pipeline
├── 04_synchronizedcapture_.md                               # Capture system details
├── 05_framemetadata_.md                                     # Metadata structure
├── 06_framesynchronizer_.md                                 # Synchronization algorithms
└── 07_cuda_stitching_pipeline_.md                          # CUDA implementation
```

### **External Dependencies**

#### **`libs/` - Third-Party Libraries**
```
libs/
├── imgui/                      # Dear ImGui library for GUI
│   ├── imgui.cpp              # Core ImGui implementation
│   ├── imgui_demo.cpp         # Example and demo code
│   ├── imgui_draw.cpp         # Rendering backend
│   ├── imgui_tables.cpp       # Table widgets
│   ├── imgui_widgets.cpp      # UI widgets
│   └── backends/              # Platform-specific backends
│       ├── imgui_impl_glfw.cpp    # GLFW window management
│       └── imgui_impl_opengl3.cpp # OpenGL rendering
│
└── gl3w/                      # OpenGL extension loader
    ├── include/GL/gl3w.h      # Header file
    └── src/gl3w.c            # Implementation
```

### **Test Data and Resources**

#### **`imgs/` - Test Images and Calibration Data**
```
imgs/
├── central.jpg                 # Sample center camera image
├── central2.jpg               # Additional center camera sample
├── derecha.jpg                # Sample right camera image  
├── derecha2.jpg               # Additional right camera sample
├── izquierda.jpg              # Sample left camera image
├── izquierda2.jpg             # Additional left camera sample
│
├── central/                   # Center camera image sequence
│   ├── image01.jpg - image07.jpg
├── derecha/                   # Right camera image sequence  
│   ├── image01.jpg - image07.jpg
└── izquierda/                 # Left camera image sequence
    ├── image01.jpg - image07.jpg
```

## 🎯 **Built Executables**

After building with CMake, you'll have these main executables:

### **1. `stitchmrsl` - Interactive GUI Application**
- **Purpose**: Visual interface for camera calibration and live stitching adjustments
- **Features**: Real-time preview, calibration adjustment, parameter tuning
- **Usage**: `./stitchmrsl`

### **2. `rtsp_stitcher` - Command-Line RTSP Processor**
- **Purpose**: Production-ready RTSP capture and stitching
- **Modes**: Capture, Process, Full Pipeline
- **Usage**: `./rtsp_stitcher <mode> <parameters>`

### **3. `video_rectifier` - Single Camera Correction Tool**
- **Purpose**: Fisheye distortion correction for pre-recorded videos
- **Features**: CUDA-accelerated batch processing
- **Usage**: `./video_rectifier <input_video> <camera_name> [output_video]`

## 📋 **Quick Start Guide**

### **1. Build the Project**
```bash
# Clone repository
git clone <repository_url>
cd stitchmrsl

# Update submodules
git submodule update --init --recursive

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### **2. Prepare Calibration**
- Place camera calibration in `intrinsic.json`
- Place camera positioning in `extrinsic.json`
- Use Python script `stitch.py` for initial calibration

### **3. Run Applications**
```bash
# Interactive GUI
./stitchmrsl

# RTSP Processing
./rtsp_stitcher capture rtsp://cam1 rtsp://cam2 rtsp://cam3

# Video Rectification  
./video_rectifier input.mp4 central output.mp4
```

## 📁 **Configuration Files**

### **Calibration Configuration**

#### **`intrinsic.json` - Camera Calibration Parameters**
Contains fisheye camera calibration data for each camera:
```json
{
  "cameras": {
    "izquierda": {   // Left camera
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],  // Camera matrix
      "dist": [[k1], [k2], [k3], [k4]],            // Distortion coefficients
      "image_size": [width, height],                // Image dimensions
      "model": "fisheye"                            // Camera model type
    },
    "central": { /* Center camera parameters */ },
    "derecha": { /* Right camera parameters */ }
  }
}
```

#### **`extrinsic.json` - Camera Positioning**
Defines transformation matrices between cameras:
```json
{
  "best_transforms": {
    "AB_similarity": [[r11, r12, tx], [r21, r22, ty], [0, 0, 1]], // Left to Center
    "BC_similarity": [[r11, r12, tx], [r21, r22, ty], [0, 0, 1]]  // Right to Center
  }
}
```

## 🔧 **System Requirements**

### **Hardware Requirements**
- **GPU**: NVIDIA with CUDA Compute Capability 6.1+
- **GPU Memory**: 4GB minimum for 1080p, 8GB recommended for 4K
- **System RAM**: 8GB minimum, 16GB recommended  
- **Storage**: SSD recommended for real-time capture
- **Network**: Gigabit Ethernet for multiple RTSP streams

### **Software Dependencies**
- **CUDA Toolkit**: 11.0 or later
- **OpenCV**: 4.5+ with CUDA support enabled
- **CMake**: 3.12 or later
- **Compiler**: GCC 7+ with C++17 support
- **Libraries**: nlohmann/json, GLFW3, OpenGL

## 🛠️ **Development Workflow**

### **Code Organization Principles**

1. **Separation of Concerns**:
   - `src/` contains implementation logic
   - `include/` contains public interfaces
   - GUI and command-line tools are separate executables

2. **CUDA Integration**:
   - GPU-accelerated processing in `cuda_stitching_pipeline.cc`
   - CPU fallback algorithms in `stitching.cc`
   - Memory management and error handling

3. **Asynchronous Design**:
   - Multi-threaded capture system
   - Non-blocking I/O operations
   - Real-time processing pipeline

### **File Naming Conventions**

- **`.cc` files**: C++ implementation
- **`.hh` files**: C++ headers
- **`.py` files**: Python scripts for calibration and testing
- **`.json` files**: Configuration and calibration data
- **`.md` files**: Documentation and guides
- **`.sh` files**: Shell scripts for automation

## 🔍 **Key Components Deep Dive**

### **RTSP Capture System**
```
synchronized_capture.cc + synchronized_capture.hh
├── Multi-threaded camera capture
├── Hardware timestamping
├── Asynchronous frame writing  
├── Memory management
└── Statistical monitoring
```

### **Frame Synchronization**
```
frame_synchronizer.cc + frame_synchronizer.hh
├── Timestamp-based alignment
├── Network jitter compensation
├── Quality-based frame selection
├── Configurable sync windows
└── Performance optimization
```

### **CUDA Processing Pipeline**
```
cuda_stitching_pipeline.cc + cuda_stitching.hh
├── GPU memory management
├── Fisheye rectification
├── Perspective warping
├── Multi-mode blending
└── Auto-cropping
```

### **GUI Interface**
```
gui.cc + gui.hh + main.cc + app.cc
├── Dear ImGui implementation
├── Real-time parameter adjustment
├── Live preview and monitoring
├── Calibration workflow
└── Performance visualization
```

## 📊 **Usage Patterns**

### **Development and Testing**
1. Use `stitch.py` for initial camera calibration
2. Test with sample images in `imgs/` directory
3. Use GUI application for parameter tuning
4. Validate with `video_rectifier` for single cameras

### **Production Deployment**
1. Use `rtsp_stitcher` for automated processing
2. Monitor performance with built-in statistics
3. Implement error handling and recovery
4. Scale with multiple GPU systems

### **Calibration Workflow**
1. Capture calibration images with fisheye patterns
2. Run Python calibration script: `python3 stitch.py`
3. Generate `intrinsic.json` and `extrinsic.json`
4. Validate calibration with GUI preview
5. Fine-tune parameters for optimal stitching

## 📚 **Additional Resources**

- **📖 [Executables Guide](EXECUTABLES_GUIDE.md)**: Detailed usage instructions for command-line tools
- **📁 [Documentation](docs/)**: Technical documentation for each component
- **🎯 Sample Data**: Test images and configurations in `imgs/` directory
- **🔧 Build Scripts**: CMake configuration for cross-platform builds

## 🤝 **Contributing**

This project follows standard C++ and Python development practices:

- **Code Style**: Follow existing formatting and naming conventions
- **Documentation**: Update relevant `.md` files for new features
- **Testing**: Validate changes with sample data in `imgs/`
- **Performance**: Monitor GPU memory usage and processing times

## 📄 **License**

[Include your license information here]
- CMake 3.12+
- C++17 compatible compiler

### **Network**
- Stable network connection to RTSP cameras
- Recommended: Gigabit Ethernet for 3x 1080p streams

## ⚙️ **Configuration**

### **Calibration Files**
- `intrinsic.json` - Camera intrinsic parameters (fisheye model)
- `extrinsic.json` - Camera extrinsic parameters and transformations

### **Sync Parameters** (configurable in code)
```cpp
// Frame synchronization settings
synchronizer.SetSyncWindow(std::chrono::milliseconds(50));    // 50ms sync window
synchronizer.SetMinSyncQuality(100.0);                       // 100ms max deviation
synchronizer.SetMaxFrameAge(std::chrono::milliseconds(200)); // 200ms max frame age
```

### **CUDA Settings**
```cpp
// GPU memory and processing settings
pipeline.SetBlendingMode(0);  // 0=max, 1=average, 2=weighted
pipeline.SetOutputSize(cv::Size(3840, 1080));  // Custom output resolution
```

## 📊 **Performance Expectations**

### **Capture Performance**
- **Input**: 3x 1080p RTSP streams @ 30 FPS
- **Capture Rate**: ~150 FPS per camera (I/O bound)
- **Memory Usage**: ~100MB circular buffers + frame queue
- **Storage Rate**: ~2.5GB/minute raw frames (PNG compressed)

### **Processing Performance** 
- **CUDA Pipeline**: ~30-60 FPS synchronized triplets
- **Memory Usage**: ~2GB GPU memory for 1080p processing
- **CPU Usage**: ~20% (mostly I/O and synchronization)
- **GPU Usage**: ~80-90% during processing phase

## 🤝 **Contributing**

This project follows standard C++ and Python development practices:

- **Code Style**: Follow existing formatting and naming conventions  
- **Documentation**: Update relevant `.md` files for new features
- **Testing**: Validate changes with sample data in `imgs/`
- **Performance**: Monitor GPU memory usage and processing times

## 📄 **License**
🐄
---

**🚀 High-Performance Multi-Camera Panoramic Stitching with CUDA Acceleration**

## For more information 
Eduardo Hernandez eduarch42@protonmail.com

David Soni sonidavid46@gmail.com
