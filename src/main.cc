#include <GL/gl3w.h>
#include "gui.hh"
#include <iostream>
#include <chrono> 

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int, char**) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.3 + GLSL 330
    const char* glsl_version = "#version 330 core";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
#endif

    // Get primary monitor and DPI scale
    GLFWmonitor* primary_monitor = glfwGetPrimaryMonitor();
    float main_scale = 1.0f;
    // We get the scale directly from GLFW instead.
    float xscale, yscale;
    glfwGetMonitorContentScale(primary_monitor, &xscale, &yscale);
    main_scale = xscale;


    // Create window with graphics context
    int window_width = (int)(1600 * main_scale);
    int window_height = (int)(1000 * main_scale);
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "OpenCV Panorama Stitching", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    if (gl3wInit() != 0) {
        std::cerr << "Failed to initialize OpenGL loader" << std::endl;
        return 1;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Application state
    ApplicationState app_state;
    
    // Register custom settings handler for manual adjustments
    RegisterManualAdjustmentsHandler(&app_state);

    // Initialize last RTSP update timestamp
    std::chrono::steady_clock::time_point last_rtsp_update = std::chrono::steady_clock::now();

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // RTSP Stream Update 
        auto now = std::chrono::steady_clock::now();
        if (app_state.use_rtsp_streams && app_state.stitching_initialized &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_rtsp_update).count() > 100) {
            app_state.UpdateRTSPFramesAndPanorama();
            last_rtsp_update = now;
        }

        // Enable docking
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

        // --- Draw GUI Panels ---
        DrawStitchingSetupPanel(app_state);
        DrawImageViewerPanel(app_state);
        DrawStatusPanel(app_state);
        DrawManualAdjustmentsPanel(app_state);
        DrawRTSPPanel(app_state);

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    app_state.ResetStitching();
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}