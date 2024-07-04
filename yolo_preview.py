#!/usr/bin/env python3

"""Show OAK camera livestream with detection model output.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on downscaled + stretched/cropped LQ frames (default: 320x320 px)
- show downscaled LQ frames + model output (bounding box, label, confidence) + fps
  in a new window (e.g. via X11 forwarding)
- optional arguments:
  '-fov' default:  stretch frames to square for model input and visualization ('-fov stretch')
                   -> full FOV is preserved, only aspect ratio is changed (adds distortion)
         optional: crop frames to square for model input and visualization ('-fov crop')
                   -> FOV is reduced due to cropping of left and right side (no distortion)
  '-af'  set auto focus range in cm (min distance, max distance)
         -> e.g. '-af 14 20' to restrict auto focus range to 14-20 cm
  '-ae'  use bounding box coordinates from detections to set auto exposure region
  '-log' print available Raspberry Pi memory, RPi CPU utilization + temperature,
         OAK memory + CPU usage and OAK chip temperature

based on open source scripts available at https://github.com/luxonis
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import depthai as dai
from apscheduler.schedulers.background import BackgroundScheduler

from utils.general import frame_norm
from utils.log import print_logs
from utils.oak_cam import bbox_set_exposure_region, set_focus_range

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-fov", "--adjust_fov", choices=["stretch", "crop"], default="stretch", type=str,
    help="Stretch frames to square ('stretch') and preserve full FOV or "
         "crop frames to square ('crop') and reduce FOV.")
parser.add_argument("-af", "--af_range", nargs=2, type=int,
    help="Set auto focus range in cm (min distance, max distance).", metavar=("CM_MIN", "CM_MAX"))
parser.add_argument("-ae", "--bbox_ae_region", action="store_true",
    help="Use bounding box coordinates from detections to set auto exposure region.")
parser.add_argument("-log", "--print_logs", action="store_true",
    help=("Print RPi available memory, RPi CPU utilization + temperature, "
          "OAK memory + CPU usage and OAK chip temperature."))
args = parser.parse_args()

# Set file paths to the detection model and corresponding config JSON
MODEL_PATH = Path("insect-detect/models/yolov5n_320_openvino_2022.1_4shave.blob")
CONFIG_PATH = Path("insect-detect/models/json/yolov5_v7_320.json")

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
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # rotate image 180°
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(320, 320)  # downscale frames for model input -> LQ frames
if args.adjust_fov == "stretch":
    cam_rgb.setPreviewKeepAspectRatio(False)  # stretch frames (16:9) to square (1:1) for model input
cam_rgb.setInterleaved(False)  # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(25)  # frames per second available for auto focus/exposure and model input

# Get sensor resolution
SENSOR_RES = cam_rgb.getResolutionSize()

# Create detection network node and define input + outputs
nn = pipeline.create(dai.node.YoloDetectionNetwork)
cam_rgb.preview.link(nn.input)  # downscaled + stretched LQ frames as model input
nn.input.setBlocking(False)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("frame")
nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

# Set detection model specific settings
nn.setBlobPath(MODEL_PATH)
nn.setNumClasses(classes)
nn.setCoordinateSize(coordinates)
nn.setAnchors(anchors)
nn.setAnchorMasks(anchor_masks)
nn.setIouThreshold(iou_threshold)
nn.setConfidenceThreshold(confidence_threshold)
nn.setNumInferenceThreads(2)

if args.af_range or args.bbox_ae_region:
    # Create XLinkIn node to send control commands to color camera node
    xin_ctrl = pipeline.create(dai.node.XLinkIn)
    xin_ctrl.setStreamName("control")
    xin_ctrl.out.link(cam_rgb.inputControl)

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    if args.print_logs:
        # Print RPi + OAK info every second
        logging.getLogger("apscheduler").setLevel(logging.WARNING)
        scheduler = BackgroundScheduler()
        scheduler.add_job(print_logs, "interval", seconds=1, id="log")
        scheduler.start()

        device.setLogLevel(dai.LogLevel.INFO)
        device.setLogOutputLevel(dai.LogLevel.INFO)

    # Create output queues to get the frames and detections from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    if args.af_range or args.bbox_ae_region:
        # Create input queue to send control commands to OAK camera
        q_ctrl = device.getInputQueue(name="control", maxSize=16, blocking=False)

    if args.af_range:
        # Set auto focus range to specified cm values
        af_ctrl = set_focus_range(args.af_range[0], args.af_range[1])
        q_ctrl.send(af_ctrl)

    # Set start time of recording and create counter to measure fps
    start_time = time.monotonic()
    counter = 0

    while True:
        # Get LQ frames + model output (detections) and show in new window together with fps
        if q_frame.has() and q_nn.has():
            frame_lq = q_frame.get().getCvFrame()
            dets = q_nn.get().detections

            counter += 1
            fps = round(counter / (time.monotonic() - start_time), 2)

            for detection in dets:
                # Get bounding box from detection model
                bbox_orig = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                bbox_norm = frame_norm(frame_lq, bbox_orig)

                # Get metadata from detection model
                label = labels[detection.label]
                det_conf = round(detection.confidence, 2)

                if args.bbox_ae_region and detection == dets[0]:
                    # Use bbox from earliest detection to set auto exposure region
                    ae_ctrl = bbox_set_exposure_region(bbox_orig, SENSOR_RES)
                    q_ctrl.send(ae_ctrl)
                    # using bbox from latest detection (dets[-1]) is also possible,
                    # but can lead to "flickering effect" in some cases

                cv2.putText(frame_lq, label, (bbox_norm[0], bbox_norm[3] + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame_lq, f"{det_conf}", (bbox_norm[0], bbox_norm[3] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.rectangle(frame_lq, (bbox_norm[0], bbox_norm[1]),
                              (bbox_norm[2], bbox_norm[3]), (0, 0, 255), 2)

            cv2.putText(frame_lq, f"fps: {fps}", (4, frame_lq.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("yolo_preview", frame_lq)

            #print(f"fps: {fps}")
            # streaming the frames via SSH (X11 forwarding) will slow down fps
            # comment out "cv2.imshow()" and print fps to console for true fps

        # Stop script and close window by pressing "Q"
        if cv2.waitKey(1) == ord("q"):
            break
