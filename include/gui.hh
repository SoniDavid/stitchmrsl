#include "app.hh"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <chrono>
#include <string>
#include <cmath>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#ifdef __EMSCRIPTEN__
#include "../libs/emscripten/emscripten_mainloop_stub.h"
#endif

void DrawStitchingInitPanel(ApplicationState& state);
void DrawAdvancedStitchingPanel(ApplicationState& state);
void DrawStitchedMainViewport(ApplicationState& state);
void DrawStitchingDebugWindow(ApplicationState& state);
void DrawMainMenuBarWithStitching(ApplicationState& state);

// Helper function to get current time in seconds
float GetTime();

// Main menu bar
void DrawMainMenuBar(ApplicationState& state);

// Camera control panel
void DrawCameraControls(ApplicationState& state);

// Main viewport window
void DrawMainViewport(ApplicationState& state);

// Stitching settings panel
void DrawStitchingPanel(ApplicationState& state);

// Performance monitoring window
void DrawPerformanceWindow(ApplicationState& state);

// Debug information window
void DrawDebugWindow(ApplicationState& state);

// Modal dialogs
void DrawModalDialogs(ApplicationState& state);