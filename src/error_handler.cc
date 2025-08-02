#include "error_handler.hh"
#include <iomanip>
#include <mutex>
#include <sstream>

// Static member definitions
std::string ErrorHandler::log_file_path_;
bool ErrorHandler::has_critical_errors_ = false;

void ErrorHandler::Log(ErrorLevel level, const std::string& component, 
                      const std::string& message, const std::string& file, int line) {
    
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    
    std::string timestamp = GetTimestamp();
    std::string level_str = LevelToString(level);
    
    // Format log message
    std::ostringstream log_stream;
    log_stream << "[" << timestamp << "] [" << level_str << "] [" << component << "] " << message;
    
    if (!file.empty() && line > 0) {
        // Extract just filename from full path
        std::string filename = file.substr(file.find_last_of("/\\") + 1);
        log_stream << " (" << filename << ":" << line << ")";
    }
    
    std::string log_line = log_stream.str();
    
    // Output to console
    if (level == ErrorLevel::ERROR || level == ErrorLevel::CRITICAL) {
        std::cerr << log_line << std::endl;
    } else {
        std::cout << log_line << std::endl;
    }
    
    // Output to log file if configured
    if (!log_file_path_.empty()) {
        std::ofstream log_file(log_file_path_, std::ios::app);
        if (log_file.is_open()) {
            log_file << log_line << std::endl;
            log_file.close();
        }
    }
    
    // Track critical errors
    if (level == ErrorLevel::CRITICAL) {
        has_critical_errors_ = true;
    }
}

void ErrorHandler::LogException(const std::exception& e, const std::string& component,
                               const std::string& context) {
    std::string message = "Exception caught: " + std::string(e.what());
    if (!context.empty()) {
        message += " (Context: " + context + ")";
    }
    
    Log(ErrorLevel::ERROR, component, message);
}

void ErrorHandler::SetLogFile(const std::string& log_file_path) {
    log_file_path_ = log_file_path;
    
    // Create initial log entry
    Log(ErrorLevel::INFO, "ErrorHandler", "Log file initialized: " + log_file_path);
}

bool ErrorHandler::HasCriticalErrors() {
    return has_critical_errors_;
}

void ErrorHandler::ResetErrorState() {
    has_critical_errors_ = false;
}

std::string ErrorHandler::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

std::string ErrorHandler::LevelToString(ErrorLevel level) {
    switch (level) {
        case ErrorLevel::INFO:     return "INFO";
        case ErrorLevel::WARNING:  return "WARN";
        case ErrorLevel::ERROR:    return "ERROR";
        case ErrorLevel::CRITICAL: return "CRITICAL";
        default:                   return "UNKNOWN";
    }
}