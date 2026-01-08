"""
Copyright (C) 2026 Videology
Programmed by Jianping Ye <jye@videologyinc.com>

Jan 08, 2026. Added pipeline json file reader and use its device_url to view rtsp stream.

"""

import argparse
import cv2

from read_json import read_pipeline, get_port_from_url

# To see whether opencv has gstreamer support.
# Need to rebuild opencv from source if not (both on Windows and on Linux).
# print(cv2.getBuildInformation())


def stream_loop(cap):
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to retrieve frame from stream.")
            break

        cv2.imshow("Scailx RTSP Stream", frame)

        # Press 'q' to exit the stream
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# gstreamer src is not supported for now ;-)
"""
gstreamer_pipeline = (
    "rtspsrc location=rtsp://scailx-ai.local:8554/stream latency=0 connection-speed=3000 ! "
    "queue ! decodebin ! queue ! videoconvert ! autovideosink sync=false"
)
"""
# cap2 = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Camera Pipeline Test", prog="opencv_stream"
    )

    parser.add_argument(
        "--port",
        "-t",
        help="Port number from rtsp server to override pipeline json file (0 = use pipeline file)",
        default="0",
    )

    parser.add_argument(
        "--pipeline",
        "-p",
        help="pipeline json file",
        default="../data/settings/camera0_pipeline.json",
    )

    parser.add_argument(
        "-i",
        "--input",
        default=1,
        type=int,
        help="Scailx camera: 1 = scailx-ai or 2 = scailx-ai-2, etc.",
    )

    args = parser.parse_args()

    pipe_dict, device_url = read_pipeline(args.pipeline)

    if device_url == "":
        device_url = "dev/video0"
    elif args.port != "0":
        # Replace device_url port
        port = get_port_from_url(device_url)
        print("old url port = ", port)
        print("user input port = ", args.port)    
        device_url = device_url.replace(":" + str(port), ":" + args.port)
        print("new url = ", device_url)

    print("Try to open device or url = ", device_url)

    scailx_rtsp_url = (
        "rtsp://scailx-ai.local:8554/stream"
        if args.input == 1
        else f"rtsp://scailx-ai-{args.input}.local:8554/stream"
    )

    # Use device url from the pipeline file for now.
    cap = cv2.VideoCapture(device_url)  # scailx_rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        exit()

    stream_loop(cap)

    cap.release()

    cv2.destroyAllWindows()
