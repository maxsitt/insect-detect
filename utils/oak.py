"""Utility functions for Luxonis OAK camera control.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    clamp(): Clamp a value between a minimum and a maximum value.
    convert_bbox_roi(): Convert bounding box coordinates to ROI (region of interest).
    convert_cm_lens_position(): Convert centimeter value to OAK lens position value.
    create_get_temp_oak(): Create a thread-safe function to get average OAK chip temperature.
    create_pipeline(): Create and configure depthai pipeline for OAK camera.
"""

import threading
from datetime import timedelta

import depthai as dai

# Create dictionary containing centimeter values and corresponding OAK lens positions
CM_LENS_POSITIONS = {
    8: 255, 9: 210, 10: 200, 11: 190, 12: 180, 13: 175, 14: 170, 15: 165, 16: 162, 17: 160,
    18: 158, 19: 156, 20: 154, 21: 152, 22: 150, 23: 148, 24: 146, 25: 144, 26: 142, 27: 141,
    28: 140, 29: 139, 30: 138, 31: 137, 32: 136, 34: 135, 36: 134, 38: 133, 40: 132, 42: 131,
    45: 130, 48: 129, 52: 128, 56: 127, 60: 126, 64: 125, 68: 124, 72: 123, 76: 122, 80: 121
}
CM_KEYS = tuple(CM_LENS_POSITIONS.keys())


def clamp(val, val_min, val_max):
    """Clamp a value between a minimum and a maximum value."""
    return max(val_min, min(val, val_max))


def convert_bbox_roi(bbox, sensor_res):
    """Convert bounding box coordinates to ROI (region of interest)."""
    x_min, y_min, x_max, y_max = [clamp(coord, 0.001, 0.999) for coord in bbox]
    roi_x, roi_y = int(x_min * sensor_res[0]), int(y_min * sensor_res[1])
    roi_w, roi_h = int((x_max - x_min) * sensor_res[0]), int((y_max - y_min) * sensor_res[1])

    return roi_x, roi_y, roi_w, roi_h


def convert_cm_lens_position(distance_cm):
    """Convert centimeter value to OAK lens position value."""
    if distance_cm in CM_KEYS:
        return CM_LENS_POSITIONS[distance_cm]

    closest_cm = min(CM_KEYS, key=lambda k: abs(k - distance_cm))
    return CM_LENS_POSITIONS[closest_cm]


def create_get_temp_oak(device):
    """Create a thread-safe function to get average OAK chip temperature."""
    temp_oak_lock = threading.Lock()

    def get_temp_oak():
        """Get average OAK chip temperature."""
        with temp_oak_lock:
            try:
                temp_oak = round(device.getChipTemperature().average)
            except RuntimeError:
                temp_oak = "NA"
        return temp_oak

    return get_temp_oak


def create_pipeline(base_path, config, config_model, use_webapp_config=False, create_xin=False):
    """Create and configure depthai pipeline for OAK camera."""
    pipeline = dai.Pipeline()

    # Get relevant config parameters from either webapp (live stream) or camera (recording) section
    config_section = getattr(config, "webapp" if use_webapp_config else "camera")
    res_hq = (config_section.resolution.width, config_section.resolution.height)      # HQ frames
    res_lq = (config.detection.resolution.width, config.detection.resolution.height)  # model input

    # Create and configure color camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setFps(config_section.fps)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    sensor_res = cam_rgb.getResolutionSize()
    cam_rgb.setInterleaved(False)  # planar layout
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    if (res_hq[0] > 1920 or res_hq[1] > 1080) and not use_webapp_config:
        if res_hq[0] <= 3840 and res_hq[1] <= 2160:
            cam_rgb.setVideoSize(*res_hq)  # crop HQ frames to configured resolution (if required)
        else:
            cam_rgb.setVideoSize(min(res_hq[0], 3840), min(res_hq[1], 2160))  # 4K limit for recording
    elif res_hq[0] > 1280 or res_hq[1] > 720:
        cam_rgb.setIspScale(1, 2)  # use ISP to downscale resolution from 4K to 1080p
        if res_hq[0] <= 1920 and res_hq[1] <= 1080:
            cam_rgb.setVideoSize(*res_hq)
        else:
            cam_rgb.setVideoSize(min(res_hq[0], 1920), min(res_hq[1], 1080))  # 1080p limit for web app
    else:
        cam_rgb.setIspScale(1, 3)  # use ISP to downscale resolution from 4K to 720p
        cam_rgb.setVideoSize(*res_hq)

    cam_rgb.setPreviewSize(*res_lq)               # downscale (+ crop) LQ frames for model input
    if abs(res_hq[0] / res_hq[1] - 1) > 0.01:     # check if HQ resolution is not ~1:1 aspect ratio
        cam_rgb.setPreviewKeepAspectRatio(False)  # stretch LQ frames to square for model input

    if config.camera.focus.mode == "manual":
        # Set manual focus position using either distance to camera (cm) or lens position (0-255)
        if config.camera.focus.distance.enabled:
            lens_pos = convert_cm_lens_position(config.camera.focus.distance.manual)
            cam_rgb.initialControl.setManualFocus(lens_pos)
        elif config.camera.focus.lens_position.enabled:
            lens_pos = config.camera.focus.lens_position.manual
            cam_rgb.initialControl.setManualFocus(lens_pos)
    elif config.camera.focus.mode == "range":
        # Set auto focus range using either distance to camera (cm) or lens position (0-255)
        if config.camera.focus.distance.enabled:
            lens_pos_min = convert_cm_lens_position(config.camera.focus.distance.range.max)
            lens_pos_max = convert_cm_lens_position(config.camera.focus.distance.range.min)
            cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)
        elif config.camera.focus.lens_position.enabled:
            lens_pos_min = config.camera.focus.lens_position.range.min
            lens_pos_max = config.camera.focus.lens_position.range.max
            cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)

    # Set ISP configuration parameters
    cam_rgb.initialControl.setSharpness(config.camera.isp.sharpness)
    cam_rgb.initialControl.setLumaDenoise(config.camera.isp.luma_denoise)
    cam_rgb.initialControl.setChromaDenoise(config.camera.isp.chroma_denoise)

    # Create and configure video encoder node and define input
    encoder = pipeline.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
    encoder.setQuality(config_section.jpeg_quality)
    cam_rgb.video.link(encoder.input)  # HQ frames as encoder input

    # Create and configure YOLO detection network node and define input
    yolo = pipeline.create(dai.node.YoloDetectionNetwork)
    yolo.setBlobPath(base_path / "models" / config.detection.model.weights)
    yolo.setConfidenceThreshold(config.detection.conf_threshold)
    yolo.setIouThreshold(config.detection.iou_threshold)
    yolo.setNumClasses(config_model.nn_config.NN_specific_metadata.classes)
    yolo.setCoordinateSize(config_model.nn_config.NN_specific_metadata.coordinates)
    yolo.setAnchors(config_model.nn_config.NN_specific_metadata.anchors)
    yolo.setAnchorMasks(config_model.nn_config.NN_specific_metadata.anchor_masks)
    yolo.setNumInferenceThreads(2)
    cam_rgb.preview.link(yolo.input)  # downscaled + cropped/stretched LQ frames as model input
    yolo.input.setBlocking(False)     # non-blocking input stream

    # Create and configure object tracker node and define inputs
    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    yolo.passthrough.link(tracker.inputTrackerFrame)  # passthrough LQ frames as tracker input
    yolo.passthrough.link(tracker.inputDetectionFrame)
    yolo.out.link(tracker.inputDetections)            # detections from YOLO model as tracker input

    # Create and configure sync node and define inputs
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=100))
    encoder.bitstream.link(sync.inputs["frames"])  # HQ frames (MJPEG-encoded bitstream)
    tracker.out.link(sync.inputs["tracker"])       # tracker + model output

    # Create message demux node and define input + outputs
    demux = pipeline.create(dai.node.MessageDemux)
    sync.out.link(demux.input)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("frame")
    demux.outputs["frames"].link(xout_rgb.input)       # synced MJPEG-encoded HQ frames

    xout_tracker = pipeline.create(dai.node.XLinkOut)
    xout_tracker.setStreamName("track")
    demux.outputs["tracker"].link(xout_tracker.input)  # synced tracker + model output

    if create_xin:
        # Create XLinkIn node to send control commands to color camera node
        xin_ctrl = pipeline.create(dai.node.XLinkIn)
        xin_ctrl.setStreamName("control")
        xin_ctrl.out.link(cam_rgb.inputControl)

    return pipeline, sensor_res
