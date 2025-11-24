
import cv2

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

# Replace with your camera's RTSP URL. 
# This often includes username, password, IP address, port, and stream path.

scailx_rtsp_url = "rtsp://scailx-ai.local:8554/stream" # cam0-gs-AR0234_1080p" 
# ip201_rtsp_url = "rtsp://10.195.53.158/rtsp_stream_00"
# ip201_rtsp_url = "rtsp://192.168.0.111/rtsp_stream_00"

gstreamer_pipeline = (
    "rtspsrc location=rtsp://scailx-ai.local:8554/stream latency=0 connection-speed=3000 ! "
    "queue ! decodebin ! queue ! videoconvert ! autovideosink sync=false"
)

# cap2 = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
cap2 = cv2.VideoCapture(scailx_rtsp_url)

if not cap2.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

stream_loop(cap2)

cap2.release()

cv2.destroyAllWindows()
