#!/usr/bin/env python

"""
Simple gstream RTSP from camera device path /dev/video0 etc.

Copyright (C) 2026 Videology
Programmed by Jianping Ye <jye@videologyinc.com>
  
Jan 026. Added gst rtsp server using camera device specified by a pipeline json file.

"""

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

from read_write_json import read_pipeline, save_pipeline, get_processing_nodes

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib


t10_ = time()
t11 = time()
t12 = time()
t13 = time()
t1_ = time()

prev_frame_time = 0
new_frame_time = 0

def get_last_node_dimension(process_list):
    n = len(process_list)
    if n>=1 and process_list[-1]["type"]=="videoscale":
        width = process_list[-1]["settings"]["width"]
        height = process_list[-1]["settings"]["height"]
        return width, height
    else:
        return 0, 0    

def get_gst_processing_string(node_dict):
    """
    Given one node dict, return its gst string.
    Now support videoscale or videocrop

    """

    if node_dict["type"]=="videoscale":
        width = node_dict["settings"]["width"]
        height = node_dict["settings"]["height"]
        return f"queue2 ! videoscale ! video/x-raw, width={width}, height={height}"
    elif node_dict["type"]=="videocrop":
        left = node_dict["settings"]["left"]
        top = node_dict["settings"]["top"]
        right = node_dict["settings"]["right"]
        bottom = node_dict["settings"]["bottom"]
        return f"queue2 ! videocrop top={top} left={left} right={right} bottom={bottom}"
    else:
        return ""

def get_gst_full_processing_string(process_list):
    """
    Given list of processing node dict, return gst full processing string.
    """

    outs = ""
    for node_dict in process_list:
        s = get_gst_processing_string(node_dict)
        if s!="":
            outs += "! " + s + " "
    return outs    


## Media factory that runs camera streaming
class StreamDataFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, pipe_dict, process_list, **properties):
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

        w, h = get_last_node_dimension(process_list)
        if w>0 and h>0:
            outwidth = w
            outheight = h
        else:
            outwidth = self.width
            outheight = self.height

        # process str is like '! ... '
        gst_processing_str = get_gst_full_processing_string(process_list)

        gst_src_string = (
            f"v4l2src device={pipe_dict['device']} "
            f"! video/x-raw,width={self.width},height={self.height},framerate={pipe_dict['fps']}/1 "
            f" {gst_processing_str} "
            f"! imxvideoconvert_g2d "
            f"! video/x-raw,width={outwidth},height={outheight},format=RGBA "
            f"! appsink"
        )

        print()
        print(gst_src_string)
        print()

        # Create opencv Video Capture
        self.cap = cv2.VideoCapture(gst_src_string, cv2.CAP_GSTREAMER,)

        # Create factory launch string
        self.launch_string = (
            f"appsrc name=source is-live=true format=GST_FORMAT_TIME "
            f"! video/x-raw,format=RGBA,width={outwidth},height={outheight},framerate={pipe_dict['fps']}/1 "
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
    def __init__(self, pipe_dict, net_url, process_list, **properties):
        super(RtspServer, self).__init__(**properties)

        # Use hostname as server mount point instead of 127.0.0.1 ;-)
        self.hostname = socket.gethostname()

        self.set_address(self.hostname)
        # Set port by user input
        self.set_service(pipe_dict["port"])

        # Create factory
        self.factory = StreamDataFactory(pipe_dict, process_list)

        # Set the factory to shared so it supports multiple clients
        self.factory.set_shared(True)

        # Add to "stream" mount point.
        # The stream will be available at rtsp://<hostname>:554/stream
        self.get_mount_points().add_factory("/stream", self.factory)
        self.attach(None)

        # Get the address
        server_address = self.get_address()
        ip_address = socket.gethostbyname(self.hostname)

        # Get the bound port number
        server_port = self.get_bound_port()
        if server_port==-1:
            # service port is not available. Set 0 using randomly assigned port instead.
            raise ValueError(f"Service port {pipe_dict['port']} is not available. Please use 0 instead to get assigned port randomly.") 

        pipe_dict["port"] = str(server_port)
        print(f"Stream URL: rtsp://{server_address}:{server_port}/stream")
        print(f"Stream URL: rtsp://{ip_address}:{server_port}/stream")


    def client_connected(self, gst_server_obj, rtsp_client_obj):
        logging.info("[INFO]: Client has connected")
        self.create_media_factories()
        self.clients_list.append(rtsp_client_obj)
        if self.verbosity > 0:
            logging.info("[INFO]: Client has connected")


def main():
    global CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y, CAPTURE_FRAMERATE, CAPTURE_DEVICE

    parser = ArgumentParser(description="gstreamer rtsp stream server")

    parser.add_argument("--port", "-t", help="Port number rtsp server sets by service (0 to set random available)", default="554")

    parser.add_argument("--pipeline", "-p", help="pipeline json file", default="data/settings/camera0_pipeline.json")
    parser.add_argument("--output", "-o", help="Output pipeline json file with any changes by user input", default="")

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
    process_list = []
    net_url = ""
    if args.pipeline is not None:
        data_dict, pipe_dict, net_url = read_pipeline(args.pipeline)
        process_list = get_processing_nodes(data_dict)

    if pipe_dict=={}:
        pipe_dict["device"] = args.device
        pipe_dict["resolution"] = args.width + "x" + args.height
        pipe_dict["fps"] = int(args.framerate)
        pipe_dict["format"] = args.format

    if net_url=="":
        net_url = f"rtsp://{socket.gethostname()}:{args.port}/stream"

    pipe_dict["port"] = args.port

    print(pipe_dict)
    if len(process_list)>0:
        print(process_list)    
    print("net url from pipeline file = ", net_url)

    Gst.init(None)
    server = RtspServer(pipe_dict, net_url, process_list)

    # Port is updated in server init if set_service("0").
    # So we need to save output pipeline json here (not before server initialized ;-)
    if args.output!="":
        save_pipeline(args.output, data_dict, pipe_dict, net_url)

    loop = GLib.MainLoop()
    loop.run()


if __name__ == "__main__":
    main()
