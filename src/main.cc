#include <GL/gl3w.h>

#include "gui.hh"
#include <iostream>

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int, char**) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(IMGUI_IMPL_OPENGL_ES3)
    const char* glsl_version = "#version 300 es";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    const char* glsl_version = "#version 330 core";  // Updated for modern OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    // Create window with graphics context and DPI-aware sizing
    GLFWmonitor* primary_monitor = glfwGetPrimaryMonitor();
    float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(primary_monitor);
    
    int window_width = (int)(1600 * main_scale);   // Increased for stitching UI
    int window_height = (int)(1000 * main_scale);  // Increased for stitching UI
    
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, 
                                         "3D Reprojection Stitching Application", nullptr, nullptr);
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
#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
#endif
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Application state with stitching integration
    ApplicationState app_state;
    app_state.dpi_scale = main_scale;

    // Print startup information
    std::cout << "=== 3D Reprojection Stitching Application ===" << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "DPI Scale: " << main_scale << "x" << std::endl;
    std::cout << "Window Size: " << window_width << "x" << window_height << std::endl;
    std::cout << "===========================================" << std::endl;

    // Main loop
#ifdef __EMSCRIPTEN__
    io.IniFilename = nullptr;
    EMSCRIPTEN_MAINLOOP_BEGIN
#else
    while (!glfwWindowShouldClose(window))
#endif
    {
        glfwPollEvents();
        
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Enable docking with correct API
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

        // Draw main menu bar with stitching integration
        if (ImGui::BeginMainMenuBar()) {
            DrawMainMenuBarWithStitching(app_state);
            ImGui::EndMainMenuBar();
        }

        // === CORE STITCHING WINDOWS ===
        
        // Stitching initialization panel
        DrawStitchingInitPanel(app_state);
        
        // Advanced stitching controls
        DrawAdvancedStitchingPanel(app_state);
        
        // Main viewport with stitched output
        DrawStitchedMainViewport(app_state);
        
        // === ORIGINAL WINDOWS (Updated) ===
        
        // Camera controls (original)
        DrawCameraControls(app_state);
        
        // Original stitching panel (now legacy/comparison)
        ImGui::Begin("Legacy Stitching Controls");
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Legacy Controls - Use Advanced Panel Instead");
        
        // Keep original controls for comparison
        const char* blend_modes[] = {"Linear", "Multiband", "Feather", "None"};
        ImGui::Combo("Legacy Blend Mode", &app_state.blend_mode, blend_modes, IM_ARRAYSIZE(blend_modes));
        ImGui::SliderFloat("Legacy Feather", &app_state.feather_size, 0.0f, 1.0f);
        ImGui::SliderFloat("Legacy A-B Weight", &app_state.ab_weight, 0.0f, 1.0f);
        ImGui::SliderFloat("Legacy B-C Weight", &app_state.bc_weight, 0.0f, 1.0f);
        
        if (ImGui::Button("Apply Legacy Settings to 3D Pipeline")) {
            if (app_state.stitching_initialized) {
                app_state.UpdateStitching();
            }
        }
        ImGui::End();
        
        // Performance monitoring
        DrawPerformanceWindow(app_state);
        
        // Debug windows
        DrawDebugWindow(app_state);
        DrawStitchingDebugWindow(app_state);
        
        // Modal dialogs
        DrawModalDialogs(app_state);

        // === 3D STITCHING UPDATE ===
        // Update stitching pipeline if auto-update is enabled
        if (app_state.stitching_initialized && app_state.auto_update_stitching) {
            // Update performance metrics based on stitching pipeline
            static auto last_update = std::chrono::high_resolution_clock::now();
            auto now = std::chrono::high_resolution_clock::now();
            auto dt = std::chrono::duration<float>(now - last_update).count();
            
            if (dt > 0.1f) { // Update every 100ms
                // Simulate stitching performance metrics
                app_state.gpu_time = 35.0f + sin(glfwGetTime()) * 10.0f;
                app_state.cpu_time = 8.0f + cos(glfwGetTime() * 1.2f) * 3.0f;
                app_state.fps = 30.0f - sin(glfwGetTime() * 0.8f) * 5.0f;
                app_state.memory_usage = 2.5f + sin(glfwGetTime() * 0.3f) * 0.5f;
                
                last_update = now;
            }
        }

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
#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_MAINLOOP_END;
#endif

    // Cleanup
    std::cout << "Shutting down 3D Reprojection Stitching Application..." << std::endl;
    
    // Application state cleanup (stitching pipeline cleanup happens in destructor)
    app_state.ResetStitching();
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "Shutdown complete." << std::endl;
    return 0;
}