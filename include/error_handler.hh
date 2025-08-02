#ifndef ERROR_HANDLER_HH
#define ERROR_HANDLER_HH

#include <iostream>
#include <string>
#include <exception>
#include <chrono>
#include <fstream>

class ErrorHandler {
public:
    enum class ErrorLevel {
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    };
    
    static void Log(ErrorLevel level, const std::string& component, 
                   const std::string& message, const std::string& file = "", 
                   int line = -1);
    
    static void LogException(const std::exception& e, const std::string& component,
                            const std::string& context = "");
    
    static void SetLogFile(const std::string& log_file_path);
    
    static bool HasCriticalErrors();
    static void ResetErrorState();

private:
    static std::string GetTimestamp();
    static std::string LevelToString(ErrorLevel level);
    
    static std::string log_file_path_;
    static bool has_critical_errors_;
};

// Convenience macros
#define LOG_INFO(component, message) \
    ErrorHandler::Log(ErrorHandler::ErrorLevel::INFO, component, message, __FILE__, __LINE__)

#define LOG_WARNING(component, message) \
    ErrorHandler::Log(ErrorHandler::ErrorLevel::WARNING, component, message, __FILE__, __LINE__)

#define LOG_ERROR(component, message) \
    ErrorHandler::Log(ErrorHandler::ErrorLevel::ERROR, component, message, __FILE__, __LINE__)

#define LOG_CRITICAL(component, message) \
    ErrorHandler::Log(ErrorHandler::ErrorLevel::CRITICAL, component, message, __FILE__, __LINE__)

#define LOG_EXCEPTION(e, component, context) \
    ErrorHandler::LogException(e, component, context)

#endif // ERROR_HANDLER_HH