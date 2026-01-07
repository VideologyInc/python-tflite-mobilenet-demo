import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer, GLib
import sys

# Initialize GStreamer
Gst.init(None)

class MyFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        super(MyFactory, self).__init__()
        self.set_shared(True)
        # Set the media pipeline description
        # This pipeline captures from a V4L2 device, encodes to H264, and payloads it for RTSP
        # Adjust 'v4l2src device=/dev/video0' and 'width,height,framerate' as needed for your camera
        self.set_launch(
            f"v4l2src device=/dev/video0 ! " 
            f"video/x-raw, width=640, height=480, framerate=30/1 ! " 
            f"videoconvert ! "
            f"x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
            f"h264parse ! "
            f"rtph264pay name=pay0 pt=96 config-interval=1 "
        )

class RTSPServer:
    def __init__(self):
        self.server = GstRtspServer.RTSPServer()
        self.factory = MyFactory()
        self.server.get_mount_points().add_factory("/test", self.factory) # Stream available at /test
        self.server.attach(None)

        print(f"RTSP server ready at rtsp://127.0.0.1:8554/test")
        print("Use a client like VLC to view the stream.")

    def run(self):
        self.loop = GLib.MainLoop()
        self.loop.run()

if __name__ == '__main__':
    rtsp_server_app = RTSPServer()
    try:
        rtsp_server_app.run()
    except KeyboardInterrupt:
        print("Server stopped by user.")
        sys.exit(0)


