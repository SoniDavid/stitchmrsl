cmake_minimum_required(VERSION 3.12)
project(stitchmrsl)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wformat")

# Define ImGui directory
set(IMGUI_DIR libs/imgui)

# GL3W setup
set(GL3W_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/gl3w")
if(EXISTS "${GL3W_DIR}/include/GL/gl3w.h")
    set(GL3W_SOURCES
        ${GL3W_DIR}/include/GL/gl3w.h
        ${GL3W_DIR}/src/gl3w.c
    )
    set(GL3W_INCLUDE_DIR ${GL3W_DIR}/include)
else()
    message(FATAL_ERROR "GL3W not found. Please generate GL3W files.")
endif()

# Source files
set(SOURCES
    src/main.cc
    src/gui.cc
    src/stitching.cc
    src/render.cc
    include/gui.hh
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    ${GL3W_SOURCES}
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(glm REQUIRED)

# FIXED: Find nlohmann/json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    pkg_check_modules(NLOHMANN_JSON nlohmann_json)
    if(NOT NLOHMANN_JSON_FOUND)
        find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp 
            PATHS /usr/include /usr/local/include /opt/homebrew/include)
        if(NOT NLOHMANN_JSON_INCLUDE_DIR)
            message(FATAL_ERROR "nlohmann/json not found. Install: sudo apt install nlohmann-json3-dev")
        endif()
    endif()
endif()

# FIXED: Add preprocessor define for GL3W
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_GL3W)

# FIXED: Include directories using target_include_directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src  # For app.hh
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${GL3W_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

# Platform-specific configurations
if(UNIX AND NOT APPLE)
    # Linux
    message(STATUS "Configuring for Linux")
    
    pkg_check_modules(GLFW3 REQUIRED glfw3)
    
    target_include_directories(${PROJECT_NAME} PRIVATE ${GLFW3_INCLUDE_DIRS})
    
    # FIXED: Link OpenCV
    target_link_libraries(${PROJECT_NAME} 
        GL
        ${GLFW3_STATIC_LIBRARIES}
        ${OpenCV_LIBS}  # Added this
    )
    
    target_compile_options(${PROJECT_NAME} PRIVATE ${GLFW3_CFLAGS_OTHER})
    
elseif(APPLE)
    # macOS
    message(STATUS "Configuring for macOS")
    
    find_library(GLFW_LIBRARY glfw PATHS /usr/local/lib /opt/local/lib /opt/homebrew/lib)
    if(NOT GLFW_LIBRARY)
        message(FATAL_ERROR "GLFW library not found")
    endif()
    
    target_include_directories(${PROJECT_NAME} PRIVATE 
        /usr/local/include 
        /opt/local/include 
        /opt/homebrew/include
    )
    
    # FIXED: Link OpenCV
    target_link_libraries(${PROJECT_NAME}
        ${OpenGL_LIBRARIES}
        ${GLFW_LIBRARY}
        ${OpenCV_LIBS}  # Added this
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
    
elseif(WIN32)
    # Windows (MinGW)
    message(STATUS "Configuring for Windows (MinGW)")
    
    pkg_check_modules(GLFW3 REQUIRED glfw3)
    
    target_include_directories(${PROJECT_NAME} PRIVATE ${GLFW3_INCLUDE_DIRS})
    
    # FIXED: Link OpenCV
    target_link_libraries(${PROJECT_NAME}
        ${OpenGL_LIBRARIES}
        glfw3
        ${OpenCV_LIBS}  # Added this
        gdi32
        imm32
    )
    
    target_compile_options(${PROJECT_NAME} PRIVATE ${GLFW3_CFLAGS_OTHER})
endif()

# Print configuration summary
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Executable: ${PROJECT_NAME}")