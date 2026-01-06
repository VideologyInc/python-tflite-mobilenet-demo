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

import json

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib


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
    def __init__(self, pipe_dict, **properties):
        super(StreamDataFactory, self).__init__(**properties)

        # Setup frame counter for timestamps
        self.number_frames = 0
        self.duration = (
            1.0 / pipe_dict["fps"]
        ) * Gst.SECOND  # duration of a frame in nanoseconds

        # set width and height from pipe_dict
        dimensions_list = pipe_dict["resolution"].split('x')

        # Convert the string elements in the list to integers
        self.width = int(dimensions_list[0])
        self.height = int(dimensions_list[1])

        # Create opencv Video Capture
        self.cap = cv2.VideoCapture(
            f"v4l2src device={pipe_dict["device"]} "
            f"! video/x-raw,width={self.width},height={self.height},framerate={pipe_dict["fps"]}/1 "
            f"! imxvideoconvert_g2d "
            f"! video/x-raw,format=BGRA "
            f"! appsink",
            cv2.CAP_GSTREAMER,
        )

        # Create factory launch string
        self.launch_string = (
            f"appsrc name=source is-live=true format=GST_FORMAT_TIME "
            f"! video/x-raw,format=BGRA,width={self.width},height={self.height},framerate={pipe_dict["fps"]}/1 "
            f"! vpuenc_h264 "
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
    def __init__(self, pipe_dict, net_url, **properties):
        super(RtspServer, self).__init__(**properties)

        # Use hostname as server mount point instead of 127.0.0.1 ;-)
        self.hostname = socket.gethostname()

        self.set_address(self.hostname)
        # Set port
        self.set_service(PORT_NUMBER)

        # Create factory
        self.factory = StreamDataFactory(pipe_dict)

        # Set the factory to shared so it supports multiple clients
        self.factory.set_shared(True)

        # Add to "stream" mount point.
        # The stream will be available at rtsp://<hostname>:554/stream
        self.get_mount_points().add_factory("/stream", self.factory)
        self.attach(None)

        # Get the address
        server_address = self.get_address()
        # Get the bound port number
        server_port = self.get_bound_port()

        print(f"Stream URL: rtsp://{server_address}:{server_port}/stream")


    def client_connected(self, gst_server_obj, rtsp_client_obj):
        logging.info("[INFO]: Client has connected")
        self.create_media_factories()
        self.clients_list.append(rtsp_client_obj)
        if self.verbosity > 0:
            logging.info("[INFO]: Client has connected")

def read_pipeline(filename):
    """
    Given input pipeline json name, parse the data section, return camera setting dict and rtsp full url.

    Arguments:
    filename -- Input pipeline json filename.
    
    Returns:
    (dict, str) -- Pair of camera setting dict and rtsp full url to access.

    """

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data_dict = json.load(file) # Deserialize the file data into a Python dictionary
            # print(data_dict)
    
        # standard Scailx Portal pipeline file should contain "inputId" and "components" / "data" section.
        if "inputId" in data_dict:
            camera_id = data_dict["inputId"]
        else:
            camera_id = "local_camera0"
        
        net_id = "input_network_stream0"    # should also add to pipeline file bottom ;-) 

        # Try to find matching "node" in components and extract its "data" section as output dict ;-)
        if "components" in data_dict:
            out_dict = {}
            out_url = ""
            for cm in data_dict["components"]:
                if "id" in cm:
                    if cm["id"]==camera_id and "data" in cm and "settings" in cm["data"]:
                        out_dict = cm["data"]["settings"]
                    if cm["id"]==net_id and "data" in cm and "settings" in cm["data"] and "location" in cm["data"]["settings"]:
                        out_url = cm["data"]["settings"]["location"]
            print("Load camera settings from pipeline json file ", filename)
            return out_dict, out_url

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return {}, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file. Check file format.")
        return {}, None

    return {}, None


def main():
    global CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y, CAPTURE_FRAMERATE, CAPTURE_DEVICE

    parser = ArgumentParser(description="gstreamer rtsp stream server")
    parser.add_argument("--pipeline", "-p", help="pipeline json file", default="data/settings/camera0_pipeline.json")

    parser.add_argument(
        "--device", "-d", help="Video device /dev/video.. ", default="/dev/video0"
    )
    parser.add_argument("--width", "-w", help="1920 or 1280 or 640 for Boson", default="1920")
    parser.add_argument("--height", help="1080 or 720 or 512 for Boson", default="1080")
    parser.add_argument(
        "--framerate", "-r", help="Capture framrate 60 or 30", default="60"
    )
    parser.add_argument(
        "--format", "-f", help="Video frame format: GREY or YUYV or RGB3 or BGR3 or NV12 etc.", default="YUYV"
    )

    args = parser.parse_args()

    print(args.device)

    pipe_dict = {}
    net_url = ""
    if args.pipeline is not None:
        pipe_dict, net_url = read_pipeline(args.pipeline)

    if pipe_dict=={}:
        pipe_dict["device"] = args.device
        pipe_dict["resolution"] = args.width + "x" + args.height
        pipe_dict["fps"] = int(args.framerate)
        pipe_dict["format"] = args.format

    if net_url=="":
        net_url = "rtsp://scailx-ai-2.local:554/stream"        

    print(pipe_dict)
    print(net_url)

    Gst.init(None)
    server = RtspServer(pipe_dict, net_url)
    loop = GLib.MainLoop()
    loop.run()


if __name__ == "__main__":
    main()
