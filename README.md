# MRSL Stitching Code

## Current Status:
As for now, this implementation is able to: 
- Take MRSL Ubiquiti G4 Pro Downwards Cameras RTSP links (Left, Central, Right)
- Rectify and Stitch the Cameras into a single image 
- Use an interactive GUI to visualize and parametrize using ImGui
- Record cameras' videos and rectify into a 15 fps video

## To do:
- Optimize performance for real time usage 
- Improve GUI implementation for a more responsive and real-time usage
- Implement cuda or other shared-resource library for fast performance
- Clean files, unused and repeated code

## To run

### 1. Clone repository and submodules
```bash
$ git clone https://github.com/SoniDavid/stitchmrsl
$ cd stitchmrsl
$ git submodule update --init --recursive
```

### 2. Run gl3w_gen.py
```bash
$ cd libs/gl3w/
$ python3 gl3w_gen.py 
```

### 3. Run CMake and Compile
On stitchmrsl directory:
```bash
$ cd build
$ cmake ..
$ make -j$(nproc)
```

### 4. Run Code
On stitchmrsl directory:
```bash 
# To run GUI
./build/stitchmrsl 

# To run video recorder
./build/rtsp_stitcher
```

