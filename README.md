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


####. Enjoy object detection :-)


***

