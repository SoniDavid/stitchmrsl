#ifndef FRAME_SYNCHRONIZER_HH
#define FRAME_SYNCHRONIZER_HH

#include "synchronized_capture.hh"
#include <vector>
#include <optional>
#include <algorithm>

struct SyncedFrameTriplet {
    std::string cam1_path, cam2_path, cam3_path;
    std::chrono::high_resolution_clock::time_point sync_timestamp;
    double sync_quality; // 0.0 = perfect sync, higher = worse sync
    uint64_t triplet_id;
    
    bool is_valid() const {
        return !cam1_path.empty() && !cam2_path.empty() && !cam3_path.empty();
    }
};

class FrameSynchronizer {
public:
    FrameSynchronizer();
    ~FrameSynchronizer();
    
    // Main synchronization function
    std::vector<SyncedFrameTriplet> FindOptimalSync(
        const std::vector<FrameMetadata>& cam1_meta,
        const std::vector<FrameMetadata>& cam2_meta,
        const std::vector<FrameMetadata>& cam3_meta
    );
    
    // Configuration
    void SetSyncWindow(std::chrono::milliseconds window);
    void SetMinSyncQuality(double min_quality);
    void SetMaxFrameAge(std::chrono::milliseconds max_age);
    
    // Statistics
    struct SyncStats {
        size_t total_triplets_found;
        size_t perfect_sync_triplets; // sync_quality < 5ms
        size_t good_sync_triplets;    // sync_quality < 16ms (1 frame at 60fps)
        double average_sync_quality;
        double best_sync_quality;
        double worst_sync_quality;
        std::chrono::milliseconds total_time_span;
        double effective_fps;
    };
    SyncStats GetSyncStats() const { return sync_stats_; }

private:
    // Find closest frame within time window
    std::optional<FrameMetadata> FindClosestFrame(
        const std::vector<FrameMetadata>& frames,
        std::chrono::high_resolution_clock::time_point target_time,
        std::chrono::milliseconds max_deviation
    );
    
    // Calculate sync quality (lower is better)
    double CalculateSyncQuality(
        const FrameMetadata& frame1,
        const FrameMetadata& frame2, 
        const FrameMetadata& frame3
    );
    
    // Remove frames that are too old or invalid
    std::vector<FrameMetadata> FilterValidFrames(
        const std::vector<FrameMetadata>& frames,
        std::chrono::high_resolution_clock::time_point reference_time
    );
    
    // Sort frames by timestamp
    void SortFramesByTimestamp(std::vector<FrameMetadata>& frames);
    
    // Configuration parameters
    std::chrono::milliseconds sync_window_;
    std::chrono::milliseconds max_frame_age_;
    double min_sync_quality_;
    
    // Statistics
    mutable SyncStats sync_stats_;
    
    // Constants
    static constexpr std::chrono::milliseconds DEFAULT_SYNC_WINDOW{33}; // ~2 frames at 60fps
    static constexpr std::chrono::milliseconds DEFAULT_MAX_FRAME_AGE{200}; // 200ms max age
    static constexpr double DEFAULT_MIN_SYNC_QUALITY = 50.0; // 50ms max deviation
    static constexpr double PERFECT_SYNC_THRESHOLD = 5.0;    // 5ms = perfect
    static constexpr double GOOD_SYNC_THRESHOLD = 16.67;     // 1 frame at 60fps
};

#endif // FRAME_SYNCHRONIZER_HH