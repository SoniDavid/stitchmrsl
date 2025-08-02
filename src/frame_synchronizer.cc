#include "frame_synchronizer.hh"
#include <iostream>
#include <algorithm>
#include <cmath>

FrameSynchronizer::FrameSynchronizer() 
    : sync_window_(DEFAULT_SYNC_WINDOW),
      max_frame_age_(DEFAULT_MAX_FRAME_AGE),
      min_sync_quality_(DEFAULT_MIN_SYNC_QUALITY) {
    // Initialize statistics
    sync_stats_ = {};
}

FrameSynchronizer::~FrameSynchronizer() = default;

std::vector<SyncedFrameTriplet> FrameSynchronizer::FindOptimalSync(
    const std::vector<FrameMetadata>& cam1_meta,
    const std::vector<FrameMetadata>& cam2_meta,
    const std::vector<FrameMetadata>& cam3_meta) {
    
    std::cout << "[SYNC] Starting frame synchronization..." << std::endl;
    std::cout << "[SYNC] Input frames: CAM1=" << cam1_meta.size() 
              << ", CAM2=" << cam2_meta.size() 
              << ", CAM3=" << cam3_meta.size() << std::endl;
    
    // Create working copies and sort by timestamp
    auto cam1_sorted = cam1_meta;
    auto cam2_sorted = cam2_meta;
    auto cam3_sorted = cam3_meta;
    
    SortFramesByTimestamp(cam1_sorted);
    SortFramesByTimestamp(cam2_sorted);
    SortFramesByTimestamp(cam3_sorted);
    
    // Find the time range that covers all cameras
    if (cam1_sorted.empty() || cam2_sorted.empty() || cam3_sorted.empty()) {
        std::cerr << "[SYNC] One or more cameras have no frames" << std::endl;
        return {};
    }
    
    auto earliest_start = std::max({
        cam1_sorted.front().timestamp,
        cam2_sorted.front().timestamp,
        cam3_sorted.front().timestamp
    });
    
    auto latest_end = std::min({
        cam1_sorted.back().timestamp,
        cam2_sorted.back().timestamp,
        cam3_sorted.back().timestamp
    });
    
    std::cout << "[SYNC] Synchronization window: " 
              << std::chrono::duration_cast<std::chrono::seconds>(latest_end - earliest_start).count() 
              << " seconds" << std::endl;
    
    // Filter frames to valid time range
    auto cam1_filtered = FilterValidFrames(cam1_sorted, earliest_start);
    auto cam2_filtered = FilterValidFrames(cam2_sorted, earliest_start);
    auto cam3_filtered = FilterValidFrames(cam3_sorted, earliest_start);
    
    std::cout << "[SYNC] Filtered frames: CAM1=" << cam1_filtered.size() 
              << ", CAM2=" << cam2_filtered.size() 
              << ", CAM3=" << cam3_filtered.size() << std::endl;
    
    std::vector<SyncedFrameTriplet> synchronized_triplets;
    std::vector<double> sync_qualities;
    
    uint64_t triplet_id = 0;
    
    // Use CAM1 as the reference camera
    for (const auto& cam1_frame : cam1_filtered) {
        // The check for max_frame_age_ was removed from post-processing
        // as it was incorrectly filtering out almost all frames.
        
        // Find closest frames in CAM2 and CAM3
        auto cam2_match = FindClosestFrame(cam2_filtered, cam1_frame.timestamp, sync_window_);
        auto cam3_match = FindClosestFrame(cam3_filtered, cam1_frame.timestamp, sync_window_);
        
        if (cam2_match.has_value() && cam3_match.has_value()) {
            // Calculate synchronization quality
            double sync_quality = CalculateSyncQuality(cam1_frame, cam2_match.value(), cam3_match.value());
            
            if (sync_quality <= min_sync_quality_) {
                SyncedFrameTriplet triplet;
                triplet.cam1_path = cam1_frame.frame_path;
                triplet.cam2_path = cam2_match.value().frame_path;
                triplet.cam3_path = cam3_match.value().frame_path;
                triplet.sync_timestamp = cam1_frame.timestamp;
                triplet.sync_quality = sync_quality;
                triplet.triplet_id = triplet_id++;
                
                synchronized_triplets.push_back(triplet);
                sync_qualities.push_back(sync_quality);
            }
        }
    }
    
    // Update statistics
    sync_stats_.total_triplets_found = synchronized_triplets.size();
    sync_stats_.perfect_sync_triplets = std::count_if(sync_qualities.begin(), sync_qualities.end(),
        [](double q) { return q < PERFECT_SYNC_THRESHOLD; });
    sync_stats_.good_sync_triplets = std::count_if(sync_qualities.begin(), sync_qualities.end(),
        [](double q) { return q < GOOD_SYNC_THRESHOLD; });
    
    if (!sync_qualities.empty()) {
        sync_stats_.average_sync_quality = std::accumulate(sync_qualities.begin(), sync_qualities.end(), 0.0) 
                                         / sync_qualities.size();
        sync_stats_.best_sync_quality = *std::min_element(sync_qualities.begin(), sync_qualities.end());
        sync_stats_.worst_sync_quality = *std::max_element(sync_qualities.begin(), sync_qualities.end());
    }
    
    sync_stats_.total_time_span = std::chrono::duration_cast<std::chrono::milliseconds>(latest_end - earliest_start);
    
    if (sync_stats_.total_time_span.count() > 0) {
        sync_stats_.effective_fps = (synchronized_triplets.size() * 1000.0) / sync_stats_.total_time_span.count();
    }
    
    // Print synchronization results
    std::cout << "\n[SYNC] SYNCHRONIZATION RESULTS:" << std::endl;
    std::cout << "[SYNC] Total synchronized triplets: " << sync_stats_.total_triplets_found << std::endl;
    std::cout << "[SYNC] Perfect sync triplets (<5ms): " << sync_stats_.perfect_sync_triplets << std::endl;
    std::cout << "[SYNC] Good sync triplets (<16.7ms): " << sync_stats_.good_sync_triplets << std::endl;
    std::cout << "[SYNC] Average sync quality: " << std::fixed << std::setprecision(2) 
              << sync_stats_.average_sync_quality << "ms" << std::endl;
    std::cout << "[SYNC] Best sync quality: " << std::fixed << std::setprecision(2) 
              << sync_stats_.best_sync_quality << "ms" << std::endl;
    std::cout << "[SYNC] Effective FPS: " << std::fixed << std::setprecision(1) 
              << sync_stats_.effective_fps << std::endl;
    
    return synchronized_triplets;
}

std::optional<FrameMetadata> FrameSynchronizer::FindClosestFrame(
    const std::vector<FrameMetadata>& frames,
    std::chrono::high_resolution_clock::time_point target_time,
    std::chrono::milliseconds max_deviation) {
    
    if (frames.empty()) {
        return std::nullopt;
    }
    
    // Binary search for closest frame
    auto it = std::lower_bound(frames.begin(), frames.end(), target_time,
        [](const FrameMetadata& frame, std::chrono::high_resolution_clock::time_point time) {
            return frame.timestamp < time;
        });
    
    std::optional<FrameMetadata> best_match;
    auto min_deviation = max_deviation;
    
    // Check frame at iterator position
    if (it != frames.end()) {
        auto time_diff = it->timestamp - target_time;
        auto deviation = std::chrono::duration_cast<std::chrono::milliseconds>(
            time_diff.count() < 0 ? -time_diff : time_diff);
            
        if (deviation <= max_deviation) {
            best_match = *it;
            min_deviation = deviation;
        }
    }
    
    // Check previous frame
    if (it != frames.begin()) {
        --it;
        auto time_diff = it->timestamp - target_time;
        auto deviation = std::chrono::duration_cast<std::chrono::milliseconds>(
            time_diff.count() < 0 ? -time_diff : time_diff);
            
        if (deviation < min_deviation) {
            best_match = *it;
        }
    }
    
    return best_match;
}

double FrameSynchronizer::CalculateSyncQuality(
    const FrameMetadata& frame1,
    const FrameMetadata& frame2,
    const FrameMetadata& frame3) {
    
    // Calculate maximum time deviation between any two frames
    auto times = {frame1.timestamp, frame2.timestamp, frame3.timestamp};
    auto min_time = *std::min_element(times.begin(), times.end());
    auto max_time = *std::max_element(times.begin(), times.end());
    
    auto max_deviation = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        max_time - min_time);
    
    return max_deviation.count();
}

std::vector<FrameMetadata> FrameSynchronizer::FilterValidFrames(
    const std::vector<FrameMetadata>& frames,
    std::chrono::high_resolution_clock::time_point reference_time) {
    
    std::vector<FrameMetadata> filtered;
    
    for (const auto& frame : frames) {
        if (frame.is_valid && frame.timestamp >= reference_time) {
            filtered.push_back(frame);
        }
    }
    
    return filtered;
}

void FrameSynchronizer::SortFramesByTimestamp(std::vector<FrameMetadata>& frames) {
    std::sort(frames.begin(), frames.end(),
        [](const FrameMetadata& a, const FrameMetadata& b) {
            return a.timestamp < b.timestamp;
        });
}

void FrameSynchronizer::SetSyncWindow(std::chrono::milliseconds window) {
    sync_window_ = window;
    std::cout << "[SYNC] Sync window set to " << window.count() << "ms" << std::endl;
}

void FrameSynchronizer::SetMinSyncQuality(double min_quality) {
    min_sync_quality_ = min_quality;
    std::cout << "[SYNC] Min sync quality set to " << min_quality << "ms" << std::endl;
}

void FrameSynchronizer::SetMaxFrameAge(std::chrono::milliseconds max_age) {
    max_frame_age_ = max_age;
    std::cout << "[SYNC] Max frame age set to " << max_age.count() << "ms" << std::endl;
}