#!/usr/bin/env python3

'''
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Website:  https://maxsitt.github.io/insect-detect-docs/
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)

This Python script does the following:
- run a custom YOLO object detection model (.blob format) on-device (Luxonis OAK)
  -> inference on downscaled LQ frames (e.g. 320x320 px)
- show downscaled LQ frames + model output (bounding box, label, confidence) + fps
  in a new window (e.g. via X11 forwarding)
- optional arguments:
  "-log" print available Raspberry Pi memory, RPi CPU utilization + temperature,
         OAK memory + CPU usage and OAK chip temperature to console
  "-ae"  use bounding box coordinates from detections to set auto exposure region

based on open source scripts available at https://github.com/luxonis
'''

import argparse
import json
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# Define optional argument
parser = argparse.ArgumentParser()
parser.add_argument("-log", "--print_logs", action="store_true",
    help="print RPi available memory, RPi CPU utilization + temperature, \
          OAK memory + CPU usage and OAK chip temperature to console")
parser.add_argument("-ae", "--bbox_ae_region", action="store_true",
    help="use bounding box coordinates from detections to set auto exposure region")
args = parser.parse_args()

if args.print_logs:
    import psutil
    from apscheduler.schedulers.background import BackgroundScheduler
    from gpiozero import CPUTemperature

# Set file paths to the detection model and config JSON
MODEL_PATH = Path("insect-detect/models/yolov5n_320_openvino_2022.1_4shave.blob")
CONFIG_PATH = Path("insect-detect/models/json/yolov5_v7_320.json")

# Get detection model metadata from config JSON
with CONFIG_PATH.open(encoding="utf-8") as f:
    config = json.load(f)
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
#cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(320, 320) # downscaled LQ frames for model input
cam_rgb.setPreviewKeepAspectRatio(False) # stretch frames (16:9) to square (1:1)
cam_rgb.setInterleaved(False) # planar layout
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(25) # frames per second available for auto focus/exposure and model input

# Get sensor resolution
sensor_res = cam_rgb.getResolutionSize()

# Create detection network node and define input + outputs
nn = pipeline.create(dai.node.YoloDetectionNetwork)
cam_rgb.preview.link(nn.input) # downscaled LQ frames as model input
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

if args.bbox_ae_region:
    # Create XLinkIn node to send control commands to OAK device
    xin_ctrl = pipeline.create(dai.node.XLinkIn)
    xin_ctrl.setStreamName("control")
    xin_ctrl.out.link(cam_rgb.inputControl)

# Define functions
def frame_norm(frame, bbox):
    """Convert relative bounding box coordinates (0-1) to pixel coordinates."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

if args.print_logs:
    def print_logs():
        """Print Raspberry Pi info to console."""
        print(f"\nAvailable RPi memory: {round(psutil.virtual_memory().available / 1048576)} MB")
        print(f"RPi CPU utilization:  {round(psutil.cpu_percent(interval=None))} %")
        print(f"RPi CPU temperature:  {round(CPUTemperature().temperature)} °C\n")

if args.bbox_ae_region:
    def bbox_set_exposure_region(xmin, ymin, xmax, ymax):
        """Use bounding box coordinates from detections to set auto exposure region."""
        if xmin < 0:
            xmin = 0.001
        if ymin < 0:
            ymin = 0.001
        if xmax > 1:
            xmax = 0.999
        if ymax > 1:
            ymax = 0.999

        ROI_x = int(xmin * sensor_res[0])
        ROI_y = int(ymin * sensor_res[1])
        ROI_width = int((xmax - xmin) * sensor_res[0])
        ROI_height = int((ymax - ymin) * sensor_res[1])

        ae_ctrl = dai.CameraControl().setAutoExposureRegion(ROI_x, ROI_y, ROI_width, ROI_height)
        q_ctrl.send(ae_ctrl)

# Connect to OAK device and start pipeline in USB2 mode
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:

    if args.print_logs:
        # Print RPi + OAK info to console every second
        scheduler = BackgroundScheduler()
        scheduler.add_job(print_logs, "interval", seconds=1, id="log")
        scheduler.start()
        device.setLogLevel(dai.LogLevel.INFO)
        device.setLogOutputLevel(dai.LogLevel.INFO)

    # Create output queues to get the frames and detections from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    if args.bbox_ae_region:
        # Create input queue to send commands to the OAK device
        q_ctrl = device.getInputQueue(name="control", maxSize=16, blocking=False)

    # Create start_time and counter variable to measure fps
    start_time = time.monotonic()
    counter = 0

    # Get LQ frames + model output (detections) and show in new window together with fps
    while True:
        if q_frame.has():
            frame = q_frame.get().getCvFrame()

            if q_nn.has():
                dets = q_nn.get().detections

                counter += 1
                fps = round(counter / (time.monotonic() - start_time), 2)

                for detection in dets:
                    # Bounding box from detection model
                    xmin, ymin = detection.xmin, detection.ymin
                    xmax, ymax = detection.xmax, detection.ymax
                    bbox = frame_norm(frame, (xmin, ymin, xmax, ymax))

                    if args.bbox_ae_region and detection == dets[0]:
                        # Use bbox from earliest detection to set exposure region
                        bbox_set_exposure_region(xmin, ymin, xmax, ymax)
                        # using bbox from latest detection (dets[-1]) is also possible,
                        # but can lead to "flickering effect" in some cases

                    # Add overlays
                    cv2.putText(frame, labels[detection.label], (bbox[0], bbox[3] + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame, f"{round(detection.confidence, 2)}", (bbox[0], bbox[3] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

                cv2.putText(frame, f"fps: {fps}", (4, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("yolo_preview", frame)

                #print(f"fps: {fps}")
                # streaming the frames via SSH (X11 forwarding) will slow down fps
                # comment out "cv2.imshow()" and print fps to console for true fps

        # Stop script and close window by pressing "Q"
        if cv2.waitKey(1) == ord("q"):
            break
