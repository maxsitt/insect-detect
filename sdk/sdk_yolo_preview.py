#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on downscaled LQ frames (e.g. 320x320 px)
- show downscaled frames + model output (bounding box, label, confidence) + fps
  in a new window (e.g. via X11 forwarding)
  -> "isp_scale=(1, 4)" will downscale 1920x1080 px to 480x270 px

based on open source scripts available at https://github.com/luxonis/depthai/tree/main/depthai_sdk
'''

from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import BboxStyle

with OakCamera(usb_speed="usb2") as oak:
#with OakCamera(usb_speed="usb2", rotation=180) as oak:
    cam_rgb = oak.camera("RGB", resolution="1080p", fps=8)
    cam_rgb.config_color_camera(isp_scale=(1, 4), interleaved=False, color_order="BGR")

    nn = oak.create_nn("/home/pi/insect-detect/models/json/yolov5_v7_320.json", input=cam_rgb) # YOLOv5n
    #nn = oak.create_nn("/home/pi/insect-detect/models/json/yolov6_v8_320.json", input=cam_rgb) # YOLOv6n
    nn.config_nn(resize_mode="stretch", conf_threshold=0.5)

    # Control auto focus and auto exposure with detections (bbox area)
    #cam_rgb.control_with_nn(nn, auto_focus=True, auto_exposure=True, debug=False)

    visualizer = oak.visualize(nn.out.main, fps=True) # downscaled frames (480x270 px)
    #visualizer = oak.visualize(nn.out.passthrough, fps=True) # downscaled + stretched frame (320x320 px)
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
    )

    oak.start(blocking=False)

    while oak.running():
        oak.poll()
