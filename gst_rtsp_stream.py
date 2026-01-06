#!/usr/bin/env python

########################################################
#
# Simple gstream RTSP from camera device path /dev/video0 etc.
#
########################################################

import sys, getopt
import numpy as np
from time import time
import os
import cv2
import gi
from argparse import ArgumentParser
import socket
import logging
import queue

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib


## Read Environment Variables

CAPTURE_DEVICE = os.environ.get("CAPTURE_DEVICE")
if not CAPTURE_DEVICE:
    CAPTURE_DEVICE = "/dev/video0"

CAPTURE_RESOLUTION_X = os.environ.get("CAPTURE_RESOLUTION_X")
if not CAPTURE_RESOLUTION_X:
    CAPTURE_RESOLUTION_X = 1920

CAPTURE_RESOLUTION_Y = os.environ.get("CAPTURE_RESOLUTION_Y")
if not CAPTURE_RESOLUTION_Y:
    CAPTURE_RESOLUTION_Y = 1080

CAPTURE_FRAMERATE = os.environ.get("CAPTURE_FRAMERATE")
if not CAPTURE_FRAMERATE:
    CAPTURE_FRAMERATE = 30

STREAM_BITRATE = os.environ.get("STREAM_BITRATE")
if not STREAM_BITRATE:
    STREAM_BITRATE = 0

PORT_NUMBER = "554"

t10_ = time()
t11 = time()
t12 = time()
t13 = time()
t1_ = time()

prev_frame_time = 0
new_frame_time = 0


## Media factory that runs camera streaming
class StreamDataFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(InferenceDataFactory, self).__init__(**properties)

        # Setup frame counter for timestamps
        self.number_frames = 0
        self.duration = (
            1.0 / CAPTURE_FRAMERATE
        ) * Gst.SECOND  # duration of a frame in nanoseconds

        # Create opencv Video Capture
        self.cap = cv2.VideoCapture(
            f"v4l2src device={DEVICE} "
            f"! video/x-raw,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 "
            f"! imxvideoconvert_g2d "
            f"! video/x-raw,format=BGRA "
            f"! appsink",
            cv2.CAP_GSTREAMER,
        )

        # Create factory launch string
        self.launch_string = (
            f"appsrc name=source is-live=true format=GST_FORMAT_TIME "
            f"! video/x-raw,format=BGRA,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 "
            f"! vpuenc_h264 bitrate={STREAM_BITRATE} "
            f"! rtph264pay config-interval=1 name=pay0 pt=96 "
        )

        print(self.launch_string)

    # Funtion to be ran for every frame that is requested for the stream
    def on_need_data(self, src, length):

        global t10_, t11, t12, t13, t1_
        global prev_frame_time, new_frame_time

        if self.cap.isOpened():

            # Read the image from the camera
            t1 = time()
            ret, image_original = self.cap.read()
            new_frame_time = time()

            fps = (
                1 / (new_frame_time - prev_frame_time)
                if new_frame_time > prev_frame_time
                else 0
            )
            prev_frame_time = new_frame_time

            if ret:
                # Resize the image to the size required for inference
                t3 = time()

                # Draw the bounding boxes for the detected objects
                img = image_original

                # Create and setup buffer
                data = GLib.Bytes.new_take(img.tobytes())
                buf = Gst.Buffer.new_wrapped_bytes(data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1

                # Emit buffer
                retval = src.emit("push-buffer", buf)
                if retval != Gst.FlowReturn.OK:
                    print(retval)
                t13 = time()
                t1_ = t1

    def get_status(self):
        print("get_status")

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def get_rtsp_media(self):
        if self.rtsp_media:
            return self.rtsp_media

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        self.rtsp_media = rtsp_media
        rtsp_media.set_reusable(True)
        appsrc = rtsp_media.get_element().get_child_by_name("source")
        appsrc.connect("need-data", self.on_need_data)

    def __del__(self):
        print("Destructor called, factory deleted.")


class RtspServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(RtspServer, self).__init__(**properties)

        # Use hostname as server mount point instead of 127.0.0.1 ;-)
        self.hostname = socket.gethostname()

        self.set_address(self.hostname)
        # Set port
        self.set_service(PORT_NUMBER)

        # Create factory
        self.factory = StreamDataFactory()

        # Set the factory to shared so it supports multiple clients
        self.factory.set_shared(True)

        # Add to "stream" mount point.
        # The stream will be available at rtsp://<hostname>:554/stream
        self.get_mount_points().add_factory("/stream", self.factory)
        self.attach(None)

    def client_connected(self, gst_server_obj, rtsp_client_obj):
        logging.info("[INFO]: Client has connected")
        self.create_media_factories()
        self.clients_list.append(rtsp_client_obj)
        if self.verbosity > 0:
            logging.info("[INFO]: Client has connected")


def main():
    global CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y, CAPTURE_FRAMERATE

    parser = ArgumentParser(description="Obeject detection - TensorFlow Lite")
    parser.add_argument(
        "--device", "-d", help="Video device /dev/video.. ", default="/dev/video0"
    )
    parser.add_argument("--resolution", "-r", help="1080p or 720p", default="1080p")
    parser.add_argument(
        "--framerate", "-f", help="Capture framrate 60 or 30", default="60"
    )

    args = parser.parse_args()

    if args.resolution == None:
        CAPTURE_RESOLUTION_X = 1920
        CAPTURE_RESOLUTION_Y = 1080
    if args.resolution == "1080p":
        CAPTURE_RESOLUTION_X = 1920
        CAPTURE_RESOLUTION_Y = 1080
    if args.resolution == "720p":
        CAPTURE_RESOLUTION_X = 1280
        CAPTURE_RESOLUTION_Y = 720

    if args.framerate == None:
        CAPTURE_FRAMERATE = 60
    else:
        CAPTURE_FRAMERATE = int(args.framerate)

    if args.object_list == None:
        OBJECT_LIST = []
    else:
        OBJECT_LIST = args.object_list

    print(CAPTURE_RESOLUTION_X, "x", CAPTURE_RESOLUTION_Y, "@", CAPTURE_FRAMERATE)

    Gst.init(None)
    server = RtspServer()
    loop = GLib.MainLoop()
    loop.run()


if __name__ == "__main__":
    main()
