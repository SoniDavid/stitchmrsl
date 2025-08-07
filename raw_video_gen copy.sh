#!/bin/bash

# Usage: ./raw_video_gen.sh <output_dir> [framerate]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_dir> [framerate]"
    exit 1
fi

OUTPUT_DIR=$1
FRAMERATE=${2:-24}  # Default framerate is 24 FPS

for CAM in cam1 cam2 cam3; do
    FRAME_DIR="$OUTPUT_DIR/raw_frames/$CAM"
    OUTPUT_VIDEO="$OUTPUT_DIR/${CAM}_output.mp4"

    if [ ! -d "$FRAME_DIR" ]; then
        echo "[WARNING] Directory not found: $FRAME_DIR"
        continue
    fi

    echo "[INFO] Generating video for $CAM..."

    ffmpeg -y -framerate $FRAMERATE -i "$FRAME_DIR/%08d.png" \
        -c:v libx264 -pix_fmt yuv420p "$OUTPUT_VIDEO"

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Video created: $OUTPUT_VIDEO"
    else
        echo "[ERROR] Failed to create video for $CAM"
    fi
done

echo "[INFO] All videos processed."
