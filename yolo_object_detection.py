#!/usr/bin/env python

# yolo_object_detection.py

########################################################
#
# WORKING OBJECT DETECTION OVER RTSP
# 1280x720@30 or 1920x1080@24 
#
########################################################

import sys, getopt
import numpy as np
from   time import time, sleep
import os
import cv2
import gi
from   argparse import ArgumentParser
import socket
import logging
import queue
import threading

from multiprocessing import Process, Pipe

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

## Import tflite runtime
import tflite_runtime.interpreter as tf

CUR_PATH = os.path.dirname(__file__)

## Read Environment Variables

USE_HW_ACCELERATED_INFERENCE = 1
MINIMUM_SCORE = 0.55
CAPTURE_DEVICE = "/dev/video0"

CAPTURE_RESOLUTION_X = 1920
CAPTURE_RESOLUTION_Y = 1080
CAPTURE_FRAMERATE = 30
STREAM_BITRATE = 0
PRINT_LOG = False

nms = True
freq = cv2.getTickFrequency()/1000   # in seconds

########################################################################################
#
# NMS implementation in Python and Numpy
#
########################################################################################
def nms_python(bboxes,pscores,threshold):
    '''
    NMS: first sort the bboxes by scores , 
        keep the bbox with highest score as reference,
        iterate through all other bboxes, 
        calculate Intersection Over Union (IOU) between reference bbox and other bbox
        if iou is greater than threshold,then discard the bbox and continue.
        
    Input:
        bboxes(numpy array of tuples) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max).
        pscores(numpy array of floats) : confidance scores for each bbox in bboxes.
        threshold(float): Overlapping threshold above which proposals will be discarded.
        
    Output:
        filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold. 
    '''
    #Unstacking Bounding Box Coordinates
    t1=time()

    bboxes = bboxes.astype('float')
    x_min = bboxes[:,0]
    y_min = bboxes[:,1]
    x_max = bboxes[:,2]
    y_max = bboxes[:,3]
    
    #Sorting the pscores in descending order and keeping respective indices.
    sorted_idx = pscores.argsort()[::-1]
    #Calculating areas of all bboxes.Adding 1 to the side values to avoid zero area bboxes.
    bbox_areas = (x_max-x_min+1)*(y_max-y_min+1)
    
    #list to keep filtered bboxes.
    filtered = []
#    print('sorted_idx=', len(sorted_idx))
    while len(sorted_idx) > 0:
        #Keeping highest pscore bbox as reference.
        rbbox_i = sorted_idx[0]
        #Appending the reference bbox index to filtered list.
        filtered.append(rbbox_i)
        
        #Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
        overlap_xmins = np.maximum(x_min[rbbox_i],x_min[sorted_idx[1:]])
        overlap_ymins = np.maximum(y_min[rbbox_i],y_min[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x_max[rbbox_i],x_max[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y_max[rbbox_i],y_max[sorted_idx[1:]])
        
        #Calculating overlap bbox widths,heights and there by areas.
        overlap_widths = np.maximum(0,(overlap_xmaxs-overlap_xmins+1))
        overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+1))
        overlap_areas = overlap_widths*overlap_heights
        
        #Calculating IOUs for all bboxes except reference bbox
        ious = overlap_areas/(bbox_areas[rbbox_i]+bbox_areas[sorted_idx[1:]]-overlap_areas)
        
        #select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > threshold)[0]+1
        delete_idx = np.concatenate(([0],delete_idx))
        
        #delete the above indices
        sorted_idx = np.delete(sorted_idx,delete_idx)
#    print('Elapsed:', (time()-t1)*1000, ' ms')    
#    print('X', bboxes[filtered])
    #Return filtered bboxes
    return bboxes[filtered].astype('int')


########################################################################################
#
# draw_bounding_boxes_with_classes
#
########################################################################################
def draw_bounding_boxes_with_classes(image, predictions, labels, scale, zero_point, confidence_threshold=0.5):
    global nms, freq

    # Iterate through the bounding box predictions
    m = 0
    n = 0

    predictions = predictions[predictions[:,4].argsort()[::-1]]

#    print("All high objectness_scores > 100: ",len(np.where(predictions[:,4] > 100)[0]))
#    p = len(np.where(predictions[:,4] > 100)[0])
#    for i in range(p+1):
#       x, y, width, height, objectness_score, *dummy = predictions[i]
#       x = (x - zero_point) * scale
#       y = (y - zero_point) * scale
#       width = (width - zero_point) * scale
#       height = (height - zero_point) * scale
#       print(predictions[i][4], predictions[i], width, height)

    t1 = cv2.getTickCount()
    bboxes  = []
    pscores = []
    labels_box = []
    boxes      = 0

    for prediction in predictions:
        x, y, width, height, objectness_score, *class_scores = prediction

        objectness_score = (objectness_score - zero_point) * scale

        m = m + 1
        if objectness_score < confidence_threshold:
            break
        n = n + 1

        # Dequantize the values
        x = (x - zero_point) * scale
        y = (y - zero_point) * scale
        width = (width - zero_point) * scale
        height = (height - zero_point) * scale
        class_scores = [(score - zero_point) * scale for score in class_scores]

        # Find the class with the highest probability
        max_class_index = np.argmax(class_scores)
        max_class_probability = class_scores[max_class_index]

        # Get the class label
        class_label = labels[max_class_index]
#        print(max_class_index, class_label, max_class_probability)

        image_height, image_width, _ = image.shape
        x1 = int((x - width / 2) * image_width)
        y1 = int((y - height / 2) * image_height)
        x2 = int((x + width / 2) * image_width)
        y2 = int((y + height / 2) * image_height)
        color = (0, 0, 255)  # Green color for the bounding box

        bboxes.append([x1, y1, x2, y2])
        pscores.append(max_class_probability)
        labels_box.append(class_label)

        if not nms:
           image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
           image = cv2.putText(image, f'{class_label} ({max_class_probability:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
           boxes = boxes + 1

    if nms:
       if len(bboxes)>0:
          bboxes_after_nms = nms_python(np.asfarray(bboxes),np.array(pscores),0.3)
          bboxes_after_nms = np.array(bboxes_after_nms)

          a = np.column_stack((bboxes, pscores, labels_box))
          b = a[a[:, 4].argsort()[::-1]]
          bboxes     = np.array(b[:,0:4], dtype = np.short)
          pscores    = np.asfarray(b[:,4])
          labels_box = np.array(b[:,5])

          _bboxes = []
          _labels_box = []
          _pscores    = []
          _bb = []
          for bbox in bboxes_after_nms:
             i = 0
             for bb in bboxes:
#                 print('# ', bbox, bb)
                 if np.array_equal(bbox, bb):
                    _bboxes.append(bb)
                    _labels_box.append(labels_box[i])
                    _pscores.append(pscores[i])
#                    print(i, bbox, bb, labels_box[i], pscores[i])
                    break
                 i = i + 1
#          print('filtered boxes')
          if len(bboxes_after_nms)!=len(_bboxes):
             print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  ERROR  <<<<<<<<<<<<<')
#          print(np.column_stack((_bboxes, _pscores, _labels_box)))
 
#          print('BBoxes_after_nms: ',len(bboxes_after_nms))
          i = 0
          for xy in _bboxes:
             image = cv2.putText(image, f'{_labels_box[i]} ({_pscores[i]:.2f})', (xy[0], xy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
             image = cv2.rectangle(image, (xy[0], xy[1]), (xy[2], xy[3]), color, 2)
             i = i + 1
          boxes = boxes + 1
#       else:
#          print('Lengte bboxes = 0')


    t2 = cv2.getTickCount()
#    print('Boxes: ', boxes, 'Elapsed time (nms + draw boxes): ', (t2-t1)/freq)
#    conn.send(image)
    return image, (t2-t1)/freq

########################################################################################
#
# crop_centered
#
########################################################################################
def crop_centered(image, target_w, target_h):
    # Get the dimensions of the input image
    height, width = image.shape[:2]

    # Calculate the coordinates of the top-left and bottom-right corners for the centered crop
    x1 = (width - target_w) // 2
    y1 = (height - target_h) // 2
    x2 = x1 + target_w
    y2 = y1 + target_h

    # Crop the image to the specified region
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image

#####################################################################################################################

prev_frame_time = 0
new_frame_time = 0
saved = False
img_ = 0

frame_buf1 = frame_buf2 = []
output_tensor2 = []
quant2 = zero_point2 = 0

########################################################################################
##
## Media factory that runs inference
##
########################################################################################
class InferenceDataFactory(GstRtspServer.RTSPMediaFactory):

    def read_frame_thread(self):
        global frame_buf1, frame_buf2

        lock = threading.Lock()
        while True:
          if self.cap.isOpened():
             lock.acquire()
             frame_buf2 = frame_buf1
             lock.release()
             if PRINT_LOG:
                print('1',cv2.getTickCount())
             _, frame_buf1 = self.cap.read()

    def invoke_thread(self):
        global frame_buf2, output_tensor2, quant2, zero_point2
        global prev_frame_time, new_frame_time

        lock = threading.Lock()
        sleep(0.1)
        while True:

           new_frame_time = time()
           fps = 1/(new_frame_time-prev_frame_time)
           prev_frame_time = new_frame_time
           t1 = time()
           image_original = np.array(frame_buf2)
#           print(image_original.size)
           if PRINT_LOG:
              print('2',cv2.getTickCount())
           crop_img = crop_centered(image_original, self.width, self.height)
           input=np.array([cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)],dtype=self.dtype)

           self.interpreter.set_tensor(self.input_details[0]['index'], input)
           self.interpreter.invoke()
           output_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])

           # Access the quantization parameters
           quantization_params = self.output_details[0]['quantization_parameters']

           # Extract scale and zero_point values
           if len(quantization_params['scales']) > 0:
              quant = quantization_params['scales'][0]  # Scale value
              zero_point = quantization_params['zero_points'][0]  # Zero-point value
           else:
              quant=0
              zero_point=0
#           lock.acquire()
           output_tensor2 = output_tensor
           quant2 = quant
           zero_point = zero_point
#           lock.release()

#           print("elapsed time: ", (time()-t1)*1000, "ips: ",  f"{fps:.5f}")

    def __init__(self, **properties):
        super(InferenceDataFactory, self).__init__(**properties)

        self.hostname = socket.gethostname()

        # Setup frame counter for timestamps
        self.number_frames = 0
        self.duration = (1.0 / CAPTURE_FRAMERATE) * Gst.SECOND  # duration of a frame in nanoseconds

        # Create opencv Video Capture
        """
        MP4_file = "highway.mp4"     
        self.cap = cv2.VideoCapture(f'filesrc location={MP4_file} ' \
                                    f'! decodebin ! video/x-raw ! queue ! videoconvert ' \
                                    f'! imxvideoconvert_g2d ' \
                                    f'! video/x-raw,format=BGRA ' \
                                    f'! appsink', cv2.CAP_GSTREAMER)

        """
        self.cap = cv2.VideoCapture(f'v4l2src name=cam_src device={DEVICE} num-buffers=-1 extra-controls="controls,horizontal_flip=0,vertical_flip=0" ' \
                                    f'! video/x-raw,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 ' \
                                    f'! imxvideoconvert_g2d ' \
                                    f'! video/x-raw,format=BGRA ' \
                                    f'! appsink', cv2.CAP_GSTREAMER)
        
        print(f'v4l2src name=cam_src device={DEVICE} num-buffers=-1 extra-controls="controls,horizontal_flip=0,vertical_flip=0" ' \
                                    f'! video/x-raw,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 ' \
                                    f'! imxvideoconvert_g2d ' \
                                    f'! video/x-raw,format=BGRA ' \
                                    f'! appsink', cv2.CAP_GSTREAMER)
                 
        delegates = [tf.load_delegate("/usr/lib/libvx_delegate.so")]
#        delegates = []

        # Load the Object Detection model and its labels
        print(MODEL_PATH)
        print(LABEL)
        with open(os.path.join(MODEL_PATH, LABEL), "r") as file:
            self.labels = file.read().splitlines()

        # Create the tensorflow-lite interpreter
        self.interpreter = tf.Interpreter(model_path=os.path.join(MODEL_PATH, MODEL),
                                          num_threads=4, experimental_delegates=delegates)

        # Allocate tensors.
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size=self.input_details[0]['shape'][1]
        print(self.input_details)

        self.width  = self.input_details[0]['shape'][1]
        self.height = self.input_details[0]['shape'][2]
        print(self.width, self.height)

        input_details  = self.interpreter.get_input_details()
        self.dtype = input_details[0]['dtype']
        
        print(self.output_details)
        # If the expected input type is int8 (quantized model), rescale data
        print(self.dtype)
        if self.dtype == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            print("Input scale:", input_scale)
            print("Input zero point:", input_zero_point)
            print()

        # Create factory launch string
        self.launch_string = f'appsrc name=source is-live=true format=GST_FORMAT_TIME ' \
                             f'! video/x-raw,format=BGRA,width={self.width},height={self.height},framerate={CAPTURE_FRAMERATE}/1 ' \
                             f'! vpuenc_h264 bitrate={STREAM_BITRATE} ' \
                             f'! rtph264pay config-interval=1 name=pay0 pt=96 '

        print(self.launch_string)

        x = threading.Thread(target=self.read_frame_thread)
        x.start()

        y = threading.Thread(target=self.invoke_thread)
        y.start()

########################################################################################
##
## Funtion to be ran for every frame that is requested for the stream
##
########################################################################################
    def on_need_data(self, src, length):
           global frame_buf2, output_tensor2, quant2, zero_point2
           
           while len(output_tensor2)==0:
              sleep(0.001)

#           print('3',cv2.getTickCount())
           crop_img = crop_centered(frame_buf2, self.width, self.height)

           output_tensor = output_tensor2
           quant = quant2
           zero_point = zero_point2

           crop_img, elapse_nms_boxes = draw_bounding_boxes_with_classes(crop_img, output_tensor[0], self.labels, scale=quant, zero_point=zero_point, confidence_threshold=0.7)

           img_ = np.array(crop_img).tobytes()

           # Create and setup buffer
           data = GLib.Bytes.new_take(img_)
           buf = Gst.Buffer.new_wrapped_bytes(data)
           buf.duration = self.duration
           timestamp = self.number_frames * self.duration
           buf.pts = buf.dts = int(timestamp)
           buf.offset = timestamp
           self.number_frames += 1

           # Emit buffer
           retval = src.emit('push-buffer', buf)
           if retval != Gst.FlowReturn.OK:
              print(retval)
           if PRINT_LOG:
              print('4',cv2.getTickCount())

    def get_status(self):
        print('get_status')

    def do_create_element(self, url):
        pipeline = Gst.parse_launch(self.launch_string)
        return pipeline

    def get_rtsp_media(self):
        if self.rtsp_media:
            return self.rtsp_media

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        self.rtsp_media = rtsp_media
        rtsp_media.set_reusable(True)
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

    def __del__(self):
        print('Destructor called, factory deleted.')

########################################################################################
##
## RtspServer
##
########################################################################################
class RtspServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(RtspServer, self).__init__(**properties)
        # Create factory
        self.factory = InferenceDataFactory()

        # Set the factory to shared so it supports multiple clients
        self.factory.set_shared(True)

        # Add to "stream" mount point. 
        # The stream will be available at rtsp://<board-ip>:8554/stream
        self.get_mount_points().add_factory("/stream", self.factory)
        self.attach(None)

    def client_connected(self, gst_server_obj, rtsp_client_obj):
        logging.info('[INFO]: Client has connected')
        self.create_media_factories()
        self.clients_list.append(rtsp_client_obj)
        if self.verbosity > 0:
            logging.info('[INFO]: Client has connected')
        
########################################################################################
##
## MAIN
##
########################################################################################
def main():
    global MODEL,MODEL_PATH,LABEL,DEVICE
    global CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y,CAPTURE_FRAMERATE
    global OBJECT_LIST

    parser = ArgumentParser(description='Obeject detection - TensorFlow Lite')
    parser.add_argument('--model_path', '-p', help='Path to model and label files', default='{}/tflite-models'.format(CUR_PATH))
#    parser.add_argument('--model',      '-m',  help='Name of model file', default='lite-model_ssd_mobilenet_v1_1_metadata_2.tflite')
#    parser.add_argument('--model',      '-m',  help='Name of model file', default='78bdea51-9cc0-453f-8af0-5daca27f1aa5.tflite')
    parser.add_argument('--model',      '-m',  help='Name of model file', default='yolov5s-int8.tflite')
#    parser.add_argument('--label',      '-l',  help='Name of label file', default='labelmap.txt')
    parser.add_argument('--label',      '-l',  help='Name of label file', default='coco.names')
    parser.add_argument('--device',     '-d', help='Video device /dev/video.. ', default='/dev/video0')
    parser.add_argument('--resolution', '-r', help='1080p or 720p', default='1080p')
    parser.add_argument('--framerate',  '-f', help='Capture framrate 60 or 30', default='30')
    parser.add_argument('--video',      '-v', help='Video mp4 input', default=False)
    parser.add_argument('--debug',      '-x', help='Print log output (default: True)', default=True)
    # NOT YET IMPLEMENTED
    parser.add_argument('--object_list', '-o', nargs='+', default=['person', 'bicycle', 'car'])

    args = parser.parse_args()

    MODEL_PATH = args.model_path    
    MODEL = args.model
    LABEL = args.label
    DEVICE = args.device
    CAPTURE_FRAMERATE = int(args.framerate)

    if args.debug=='False':
        PRINT_LOG = False
    else:
        PRINT_LOG = True

    if args.resolution == None:
        CAPTURE_RESOLUTION_X=1920
        CAPTURE_RESOLUTION_Y=1080
    if args.resolution == '1080p':
        CAPTURE_RESOLUTION_X=1920
        CAPTURE_RESOLUTION_Y=1080
    if args.resolution == '720p':
        CAPTURE_RESOLUTION_X=1280
        CAPTURE_RESOLUTION_Y=720

    OBJECT_LIST = args.object_list

    print(os.path.join(MODEL_PATH, MODEL))
    print(os.path.join(MODEL_PATH, LABEL))
    print(CAPTURE_RESOLUTION_X,'x',CAPTURE_RESOLUTION_Y,'@',CAPTURE_FRAMERATE)

    Gst.init(None)
    server = RtspServer()
    loop = GLib.MainLoop()
    loop.run()


if __name__ == "__main__":
    main()
