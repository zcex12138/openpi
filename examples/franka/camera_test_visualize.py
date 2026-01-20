"""Test script for visualizing camera frames using qtcv."""

from __future__ import annotations

import argparse
import time

import cv2
from xensesdk.ezgl.utils.QtTools import qtcv

from examples.franka.camera_client import CameraClient


def main():
    parser = argparse.ArgumentParser(description="Visualize camera frames from camera service")
    parser.add_argument("--host", type=str, default="localhost", help="Camera service host")
    parser.add_argument("--port", type=int, default=5050, help="Camera service port")
    parser.add_argument("--timeout", type=float, default=1.0, help="Socket timeout in seconds")
    args = parser.parse_args()

    print(f"Connecting to camera service at {args.host}:{args.port}...")

    with CameraClient(args.host, args.port, timeout_s=args.timeout) as client:
        # Test connection
        if not client.ping():
            print("Failed to connect to camera service")
            return

        print("Connected! Press 'q' to quit.")

        while True:
            try:
                frames, _, _ = client.get_frames()

                # Get RGB frames from both cameras
                l500_rgb = frames.get("l500_rgb")
                d400_rgb = frames.get("d400_rgb")

                # Display L500 RGB frame (convert RGB to BGR for correct color display)
                if l500_rgb is not None:
                    l500_bgr = l500_rgb[:, :, ::-1].copy()
                    # Draw red points on L500 frame (x, y)
                    for x, y in [(178, 93), (155, 331), (716, 299), (945, 285), (863, 90), (476, 212)]:
                        cv2.circle(l500_bgr, (x, y), 4, (0, 0, 255), -1)
                    qtcv.imshow("L500 RGB Frame", l500_bgr)

                # Display D400 RGB frame (convert RGB to BGR for correct color display)
                if d400_rgb is not None:
                    d400_bgr = d400_rgb[:, :, ::-1].copy()
                    qtcv.imshow("D400 RGB Frame", d400_bgr)

                # Check for quit key
                key = qtcv.waitKey(1)
                if key is not None and (key & 0xFF) == ord("q"):
                    print("Quit requested.")
                    break

            except Exception as e:
                print(f"Error getting frames: {e}")
                time.sleep(0.1)

    print("Done.")


if __name__ == "__main__":
    main()
