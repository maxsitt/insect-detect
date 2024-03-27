#!/usr/bin/env python3

"""Show OAK camera livestream with detection model and object tracker output.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on downscaled + stretched LQ frames (default: 320x320 px)
- use an object tracker to track detected objects and assign unique tracking IDs
  -> accuracy depends on object motion speed and inference speed of the detection model
- show downscaled frames + model/tracker output (bounding box, label, confidence,
  tracklet trail) + fps in a new window (e.g. via X11 forwarding)
  -> 'isp_scale=(1, 4)' will downscale 1920x1080 px to 480x270 px

based on open source scripts available at https://github.com/luxonis
"""

import depthai as dai
from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import BboxStyle

with OakCamera(usb_speed="usb2") as oak:
#with OakCamera(usb_speed="usb2", rotation=180) as oak:  # rotate image 180°
    cam_rgb = oak.camera("RGB", resolution="1080p", fps=8)
    cam_rgb.config_color_camera(isp_scale=(1, 4), interleaved=False, color_order="BGR")

    nn = oak.create_nn("/home/pi/insect-detect/models/json/yolov5_v7_320.json", input=cam_rgb, tracker=True)  # YOLOv5n
    #nn = oak.create_nn("/home/pi/insect-detect/models/json/yolov6_v8_320.json", input=cam_rgb, tracker=True) # YOLOv6n
    nn.config_nn(resize_mode="stretch", conf_threshold=0.5)

    nn.config_tracker(
        tracker_type=dai.TrackerType.ZERO_TERM_IMAGELESS,
        #tracker_type=dai.TrackerType.SHORT_TERM_IMAGELESS,  # better for low fps
        assignment_policy=dai.TrackerIdAssignmentPolicy.UNIQUE_ID,
        threshold=0.1
    )

    # Use bounding box coordinates from detections to set auto exposure region
    #cam_rgb.control_with_nn(nn, auto_focus=False, auto_exposure=True, debug=False)

    visualizer = oak.visualize(nn.out.tracker, fps=True)  # downscaled frames (480x270 px)
    visualizer.detections(
        bbox_style=BboxStyle.ROUNDED_CORNERS,
        #bbox_style=BboxStyle.RECTANGLE,
        fill_transparency=0
    ).text(
        font_face=0,
        font_color=(255, 255, 255),
        font_thickness=1,
        font_scale=0.5,
        auto_scale=False
    ).tracking(
        max_length=300,   # tracklet trail length
        line_thickness=1  # tracklet trail thickness
    )

    oak.start(blocking=False)

    while oak.running():
        oak.poll()
