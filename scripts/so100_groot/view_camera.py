#!/usr/bin/env python3
"""
Simple camera viewer script for adjusting camera positions.

Usage:
    python view_camera.py /dev/scene
    python view_camera.py /dev/wrist
    python view_camera.py /dev/scene --width 1280 --height 720
"""

import argparse
import cv2
import sys


def view_camera(device_path, width=640, height=480):
    """
    Open a camera preview window.
    
    Args:
        device_path: Camera device path (e.g., /dev/scene, /dev/wrist)
        width: Frame width in pixels
        height: Frame height in pixels
    """
    print(f"Opening camera: {device_path}")
    print(f"Resolution: {width}x{height}")
    print("Press 'q' to quit")
    print("-" * 50)
    
    # Open camera
    cap = cv2.VideoCapture(device_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera at {device_path}")
        print("\nAvailable devices:")
        import os
        devices = [f for f in os.listdir('/dev') if f.startswith('video') or f in ['scene', 'wrist']]
        for dev in sorted(devices):
            print(f"  /dev/{dev}")
        sys.exit(1)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Get actual resolution (might differ from requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Actual resolution: {actual_width}x{actual_height}")
    
    window_name = f"{device_path} - {actual_width}x{actual_height}"
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            # Add text overlay with info
            cv2.putText(frame, f"{device_path}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"{actual_width}x{actual_height}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, actual_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(window_name, frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed")


def main():
    parser = argparse.ArgumentParser(
        description="View camera feed for adjusting position",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_camera.py /dev/scene
  python view_camera.py /dev/wrist
  python view_camera.py /dev/scene --width 1280 --height 720
  python view_camera.py /dev/wrist --width 640 --height 480
        """
    )
    
    parser.add_argument(
        "device",
        help="Camera device path (e.g., /dev/scene, /dev/wrist, /dev/video0)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width in pixels (default: 640)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height in pixels (default: 480)"
    )
    
    args = parser.parse_args()
    
    view_camera(args.device, args.width, args.height)


if __name__ == "__main__":
    main()

