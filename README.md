## Run AI models to do object detection with Scailx camera video stream.

***

#### 1. Follow Scailx document online section 3.4 for main steps to run python programs on camera to do live stream object detection.

#### https://videology-inc.atlassian.net/wiki/spaces/SUD/pages/63275010/Running+AI+models+in+Python+with+Tensorflow-lite

#### On camera, please use git link to clone the main branch from the repository instead of https after git lfs install.

#### git clone git@github.com:VideologyInc/python-tflite-mobilenet-demo.git

#### 2. Make sure to stop go2rtc service before running any python programs to avoid conflict and restart go2rtc service after python programs finished.

#### List services using port 8554:  `lsof -i :8554`
	
#### Stop go2rtc service:  		`systemctl stop go2rtc.service`

#### Run python program to do object detection (see below) ...

#### Restart go2rtc service to make camera stream accessible again on Windows web or VLC player afterwards.
#### 	`systemctl restart go2rtc.service`

#### 3. The 2 python programs have different pros and cons. Please test running both of them with various camera settings.

#### On camera, run

#### `python3 object-detection.py -d /dev/video0`

#### Or.
#### `python3 yolo_object-detection.py -d /dev/video0	`

#### 4. To see object detction effective on Windows, please run

	`gst-launch-1.0 rtspsrc location=rtsp://scailx-ai.local:8554/stream latency=0 connection-speed=3000 ! queue ! decodebin ! queue ! videoconvert ! autovideosink sync=false`

=================================================================

#### 5. Alternatively, on Windows and Linux host (such as in VirtualBox guest systems), we can run the OpenCV python program ~/host/opencv_stream.py to access camera object detection stream to avoid complex gstreamer command in step 4.

#### 5.1	On Windows python >=3.10 venv.

#### `python opencv_stream.py`    

#### 5.2	On VirtualBox Ubuntu guest system with python >=3.10 venv.

####	Because our laptop have 2 network routes - one to company's network (10.* ip address) and one to engineering lap (ip address 192.168.*), while Scailx camera is connected to engineering network (ip address 192.168.*), in VirtualBox we first need to enable webcamera=>find Scailx camera (>=2nd on the list, 1st is usually our Laptop camera or web camera ;-) for the guest ubuntu system, then replace the rtsp line in the python program hostname using ip address of our Scailx camera, for example
	
####    scailx_rtsp_url = "rtsp://192.168.9.31:8554/stream"

####	Now run the same command in Ubuntu python venv. (Make sure we can see the camera device with `ls -l /dev/video*`).

#### `python opencv_stream.py`    

#### Enjoy object detection :-)

***

