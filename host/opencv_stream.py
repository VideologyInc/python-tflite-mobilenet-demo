import argparse
import cv2

from read_json import read_pipeline

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

    parser.add_argument("--pipeline", "-p", help="pipeline json file", default="usbcamera_pipeline.json")

    parser.add_argument(
        "-i",
        "--input",
        default=1,
        type=int,
        help="Scailx camera: 1 = scailx-ai or 2 = scailx-ai-2",
    )

    args = parser.parse_args()

    pipe_dict, device_url = read_pipeline(args.pipeline)	

    if device_url=="":
        device_url= "dev/video0"
        	
    scailx_rtsp_url = (
        "rtsp://scailx-ai.local:8554/stream"
        if args.input == 1
        else "rtsp://scailx-ai-2.local:8554/stream"
    )

    cap = cv2.VideoCapture(device_url) # scailx_rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        exit()

    stream_loop(cap)

    cap.release()

    cv2.destroyAllWindows()
