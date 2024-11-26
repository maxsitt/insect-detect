#!/usr/bin/env python3

"""Show OAK camera livestream with detection model output.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on downscaled + stretched/cropped LQ frames (default: 320x320 px)
- show downscaled LQ frames + model output (bounding box, label, confidence) + FPS
  in a new window (e.g. via X11 forwarding)
- optional arguments:
  '-fov' default:  stretch frames to square for model input and visualization ('-fov stretch')
                   -> FOV is preserved, only aspect ratio of LQ frames is changed (adds distortion)
         optional: crop frames to square for model input and visualization ('-fov crop')
                   -> FOV is reduced due to cropping of LQ frames (no distortion)
  '-af'  set auto focus range in cm (min - max distance to camera)
         -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm
  '-mf'  set manual focus position in cm (distance to camera)
         -> e.g. '-mf 14' to set manual focus position to 14 cm
  '-ae'  use bounding box coordinates from detections to set auto exposure region
  '-log' print available Raspberry Pi memory (MB), RPi CPU utilization (%) + temperature,
         OAK memory + CPU usage and OAK chip temperature at specified interval (default: 1 s)

based on open source scripts available at https://github.com/luxonis
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler

from utils.log import print_logs
from utils.oak_cam import convert_bbox_roi, convert_cm_lens_position

# Define optional arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument("-fov", "--field_of_view", type=str, choices=["stretch", "crop"], default="stretch",
    help=("Stretch frames to square and preserve FOV ('stretch') or "
          "crop frames to square and reduce FOV ('crop') (default: 'stretch')."))
group.add_argument("-af", "--auto_focus_range", type=int, nargs=2, metavar=("CM_MIN", "CM_MAX"),
    help="Set auto focus range in cm (min - max distance to camera).")
group.add_argument("-mf", "--manual_focus", type=int, metavar="CM",
    help="Set manual focus position in cm (distance to camera).")
parser.add_argument("-ae", "--auto_exposure_region", action="store_true",
    help="Use bounding box coordinates from detections to set auto exposure region.")
parser.add_argument("-log", "--print_logs", action="store_true",
    help=("Print RPi available memory (MB), RPi CPU utilization (%%) + temperature, "
          "OAK memory + CPU usage and OAK chip temperature."))
args = parser.parse_args()

# Set file paths to the detection model and corresponding config JSON
MODEL_PATH = Path.home() / "insect-detect" / "models" / "yolov5n_320_openvino_2022.1_4shave.blob"
CONFIG_PATH = Path.home() / "insect-detect" / "models" / "json" / "yolov5_v7_320.json"

# Set camera frame rate
FPS = 20  # default: 20 FPS

# Set time interval at which RPi logs are printed if "-log" is used
LOG_INT = 1  # default: 1 second

# Get detection model metadata from config JSON
with CONFIG_PATH.open(encoding="utf-8") as config_json:
    config = json.load(config_json)
nn_config = config.get("nn_config", {})
nn_metadata = nn_config.get("NN_specific_metadata", {})
classes = nn_metadata.get("classes", {})
coordinates = nn_metadata.get("coordinates", {})
anchors = nn_metadata.get("anchors", {})
anchor_masks = nn_metadata.get("anchor_masks", {})
iou_threshold = nn_metadata.get("iou_threshold", {})
confidence_threshold = nn_metadata.get("confidence_threshold", {})
nn_mappings = config.get("mappings", {})
labels = nn_mappings.get("labels", {})

# Create depthai pipeline
pipeline = dai.Pipeline()

# Create and configure color camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setFps(FPS)  # frames per second available for auto focus/exposure and model input
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(320, 320)  # downscale frames for model input -> LQ frames (1:1)
if args.field_of_view == "stretch":
    cam_rgb.setPreviewKeepAspectRatio(False)  # stretch LQ frames to square for model input
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
SENSOR_RES = cam_rgb.getResolutionSize()
img_width, img_height = cam_rgb.getPreviewSize()
norm_vals = np.array([img_width, img_height, img_width, img_height])  # used for bbox conversion

if args.auto_focus_range:
    # Convert cm to lens position values and set auto focus range
    lens_pos_min = convert_cm_lens_position(args.auto_focus_range[1])
    lens_pos_max = convert_cm_lens_position(args.auto_focus_range[0])
    cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)
elif args.manual_focus:
    # Convert cm to lens position value and set manual focus position
    lens_pos = convert_cm_lens_position(args.manual_focus)
    cam_rgb.initialControl.setManualFocus(lens_pos)

# Create and configure YOLO detection network node and define input + outputs
yolo = pipeline.create(dai.node.YoloDetectionNetwork)
yolo.setBlobPath(MODEL_PATH)
yolo.setNumClasses(classes)
yolo.setCoordinateSize(coordinates)
yolo.setAnchors(anchors)
yolo.setAnchorMasks(anchor_masks)
yolo.setIouThreshold(iou_threshold)
yolo.setConfidenceThreshold(confidence_threshold)
yolo.setNumInferenceThreads(2)
cam_rgb.preview.link(yolo.input)  # downscaled + stretched/cropped LQ frames as model input
yolo.input.setBlocking(False)     # non-blocking input stream

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
yolo.passthrough.link(xout_rgb.input)  # passthrough LQ frames for visualization

xout_yolo = pipeline.create(dai.node.XLinkOut)
xout_yolo.setStreamName("yolo")
yolo.out.link(xout_yolo.input)  # model output

if args.auto_exposure_region:
    # Create XLinkIn node to send control commands to color camera node
    xin_ctrl = pipeline.create(dai.node.XLinkIn)
    xin_ctrl.setStreamName("control")
    xin_ctrl.out.link(cam_rgb.inputControl)

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    # Create output queues to get the LQ frames and model output
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_yolo = device.getOutputQueue(name="yolo", maxSize=4, blocking=False)

    if args.auto_exposure_region:
        # Create input queue to send control commands to OAK camera
        q_ctrl = device.getInputQueue(name="control", maxSize=4, blocking=False)

    if args.print_logs:
        # Print RPi + OAK info at specified interval
        logging.getLogger("apscheduler").setLevel(logging.WARNING)  # decrease apscheduler logging level
        scheduler = BackgroundScheduler()
        scheduler.add_job(print_logs, "interval", seconds=LOG_INT, id="log")
        scheduler.start()
        device.setLogLevel(dai.LogLevel.INFO)
        device.setLogOutputLevel(dai.LogLevel.INFO)

    # Set start time of recording and create counter to measure FPS
    start_time = time.monotonic()
    counter = 0

    while True:
        if q_frame.has():
            # Get LQ frame and show in new window together with FPS
            frame_lq = q_frame.get().getCvFrame()

            counter += 1
            fps = round(counter / (time.monotonic() - start_time), 2)

            if q_yolo.has():
                # Get model output
                detections = q_yolo.get().detections
                for detection in detections:
                    # Get bounding box from model output
                    bbox_norm = (detection.xmin, detection.ymin,
                                 detection.xmax, detection.ymax)  # normalized bounding box
                    bbox = (np.clip(bbox_norm, 0, 1) * norm_vals).astype(int)  # convert to pixel coordinates

                    # Get metadata from model output
                    label = labels[detection.label]
                    confidence = round(detection.confidence, 2)

                    if args.auto_exposure_region and detection is detections[0]:
                        # Use model bbox from earliest detection to set auto exposure region
                        roi_x, roi_y, roi_w, roi_h = convert_bbox_roi(bbox_norm, SENSOR_RES)
                        q_ctrl.send(dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_w, roi_h))

                    cv2.putText(frame_lq, f"{label}", (bbox[0], bbox[3] + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame_lq, f"{confidence}", (bbox[0], bbox[3] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.rectangle(frame_lq, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            cv2.putText(frame_lq, f"FPS: {fps}", (4, img_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("yolo_preview", frame_lq)

            #print(f"FPS: {fps}")
            # streaming the frames via SSH (X11 forwarding) will slow down FPS
            # comment out "cv2.imshow()" and print FPS to console for true FPS

        # Stop script and close window by pressing "Q"
        if cv2.waitKey(1) == ord("q"):
            break
