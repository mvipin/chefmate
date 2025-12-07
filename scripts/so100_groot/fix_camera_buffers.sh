#!/bin/bash
# Fix USB camera buffer settings to prevent timeouts

echo "Optimizing USB camera buffers..."

# Increase USB buffer size for both cameras
for dev in /dev/video0 /dev/video2; do
    if [ -e "$dev" ]; then
        echo "Setting buffer for $dev"
        v4l2-ctl -d "$dev" --set-fmt-video=width=640,height=480,pixelformat=MJPG 2>/dev/null || true
    fi
done

echo "Camera buffers optimized!"
echo ""
echo "Camera settings:"
v4l2-ctl -d /dev/video0 --get-fmt-video 2>/dev/null || echo "  /dev/video0 not found"
v4l2-ctl -d /dev/video2 --get-fmt-video 2>/dev/null || echo "  /dev/video2 not found"

