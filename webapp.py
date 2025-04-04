"""Stream OAK camera live feed and configure settings via NiceGUI-based web app.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Run this script with the Python interpreter from the virtual environment where you installed
the required packages, e.g. with 'env_insdet/bin/python3 insect-detect/webapp.py'.

Modify the 'configs/config_selector.yaml' file to select the active configuration file
that will be used to load all configuration parameters.

- load YAML file with configuration parameters and JSON file with detection model parameters
- stream frames (MJPEG-encoded bitstream) from OAK camera to browser-based web app via HTTP
- draw SVG overlay with tracker/model data on frames (bounding box, label, confidence, tracking ID)
- control camera settings via web app
- save modified configuration parameters to config file

partly based on scripts from https://github.com/luxonis and https://github.com/zauberzeug/nicegui
"""

import asyncio
import base64
import copy
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import depthai as dai
import ruamel.yaml
from fastapi import Response
from nicegui import Client, app, core, ui

from utils.config import parse_json, parse_yaml
from utils.oak import convert_bbox_roi, convert_cm_lens_position

# Set camera trap ID (default: hostname) and base path (default: "insect-detect" directory)
CAM_ID = socket.gethostname()
BASE_PATH = Path.home() / "insect-detect"


def convert_duration(total_minutes):
    """Convert minutes to a dictionary with hours, minutes, and total values."""
    return {"hours": total_minutes // 60, "minutes": total_minutes % 60, "total": total_minutes}


def update_nested_dict(template, updates, defaults):
    """Update nested dictionary recursively. Replace 'None' with default values."""
    for key, value in updates.items():
        if (isinstance(value, dict) and 
            isinstance(template.get(key), dict) and 
            isinstance(defaults.get(key), dict)):
            update_nested_dict(template[key], value, defaults[key])
        else:
            template[key] = value if value is not None else defaults[key]


def update_config_selector(config_active):
    """Update the config selector file to point to the active configuration."""
    yaml = ruamel.yaml.YAML()
    yaml.width = 150  # maximum line width before wrapping
    yaml.preserve_quotes = True  # preserve all comments

    config_selector_path = BASE_PATH / "configs" / "config_selector.yaml"

    with open(config_selector_path, "r", encoding="utf-8") as file:
        config_selector = yaml.load(file)

    config_selector["config_active"] = config_active

    with open(config_selector_path, "w", encoding="utf-8") as file:
        yaml.dump(config_selector, file)


def create_pipeline(config, config_model):
    """Create and configure depthai pipeline for OAK camera."""
    pipeline = dai.Pipeline()

    # Create and configure color camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setFps(config.webapp.fps)  # frames per second available for focus/exposure and model input
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    SENSOR_RES = cam_rgb.getResolutionSize()
    cam_rgb.setInterleaved(False)  # planar layout
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    RES_HQ = (config.webapp.resolution.width, config.webapp.resolution.height)        # HQ stream
    RES_LQ = (config.detection.resolution.width, config.detection.resolution.height)  # model input

    if RES_HQ[0] > 1280 or RES_HQ[1] > 720:
        cam_rgb.setIspScale(1, 2)             # use ISP to downscale 4K to 1080p resolution
        if RES_HQ[0] < 1920 or RES_HQ[1] < 1080:
            cam_rgb.setVideoSize(*RES_HQ)     # crop to configured HQ resolution
        else:
            cam_rgb.setVideoSize(1920, 1080)  # always cap resolution at 1080p
    else:
        cam_rgb.setIspScale(1, 3)             # use ISP to downscale 4K to 720p resolution
        if RES_HQ != (1280, 720):
            cam_rgb.setVideoSize(*RES_HQ)     # crop to configured HQ resolution

    cam_rgb.setPreviewSize(*RES_LQ)               # downscale frames for model input -> LQ frames
    if abs(RES_HQ[0] / RES_HQ[1] - 1) > 0.01:     # check if HQ resolution is not ~1:1 aspect ratio
        cam_rgb.setPreviewKeepAspectRatio(False)  # stretch LQ frames to square for model input

    if config.camera.focus.mode == "range":
        # Set auto focus range using either distance to camera (cm) or lens position (0-255)
        if config.camera.focus.distance.enabled:
            lens_pos_min = convert_cm_lens_position(config.camera.focus.distance.range.max)
            lens_pos_max = convert_cm_lens_position(config.camera.focus.distance.range.min)
            cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)
        elif config.camera.focus.lens_position.enabled:
            lens_pos_min = config.camera.focus.lens_position.range.min
            lens_pos_max = config.camera.focus.lens_position.range.max
            cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)
    elif config.camera.focus.mode == "manual":
        # Set manual focus position using either distance to camera (cm) or lens position (0-255)
        if config.camera.focus.distance.enabled:
            lens_pos = convert_cm_lens_position(config.camera.focus.distance.manual)
            cam_rgb.initialControl.setManualFocus(lens_pos)
        elif config.camera.focus.lens_position.enabled:
            lens_pos = config.camera.focus.lens_position.manual
            cam_rgb.initialControl.setManualFocus(lens_pos)

    # Set ISP configuration parameters
    cam_rgb.initialControl.setSharpness(config.camera.isp.sharpness)
    cam_rgb.initialControl.setLumaDenoise(config.camera.isp.luma_denoise)
    cam_rgb.initialControl.setChromaDenoise(config.camera.isp.chroma_denoise)

    # Create and configure video encoder node and define input
    encoder = pipeline.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
    encoder.setQuality(config.webapp.jpeg_quality)
    cam_rgb.video.link(encoder.input)  # HQ frames as encoder input

    # Create and configure YOLO detection network node and define input
    yolo = pipeline.create(dai.node.YoloDetectionNetwork)
    yolo.setBlobPath(BASE_PATH / "models" / config.detection.model.weights)
    yolo.setConfidenceThreshold(config.detection.conf_threshold)
    yolo.setIouThreshold(config.detection.iou_threshold)
    yolo.setNumClasses(config_model.nn_config.NN_specific_metadata.classes)
    yolo.setCoordinateSize(config_model.nn_config.NN_specific_metadata.coordinates)
    yolo.setAnchors(config_model.nn_config.NN_specific_metadata.anchors)
    yolo.setAnchorMasks(config_model.nn_config.NN_specific_metadata.anchor_masks)
    yolo.setNumInferenceThreads(2)
    cam_rgb.preview.link(yolo.input)  # downscaled + stretched/cropped LQ frames as model input
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
    encoder.bitstream.link(sync.inputs["frames"])
    tracker.out.link(sync.inputs["tracker"])

    # Create message demux node and define input + outputs
    demux = pipeline.create(dai.node.MessageDemux)
    sync.out.link(demux.input)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("frame")
    demux.outputs["frames"].link(xout_rgb.input)

    xout_tracker = pipeline.create(dai.node.XLinkOut)
    xout_tracker.setStreamName("track")
    demux.outputs["tracker"].link(xout_tracker.input)

    # Create XLinkIn node to send control commands to color camera node
    xin_ctrl = pipeline.create(dai.node.XLinkIn)
    xin_ctrl.setStreamName("control")
    xin_ctrl.out.link(cam_rgb.inputControl)

    return pipeline, SENSOR_RES


async def start_camera():
    """Start OAK camera with selected configuration."""

    # Parse active config file and load configuration parameters
    app.state.config_selector = parse_yaml(BASE_PATH / "configs" / "config_selector.yaml")
    app.state.config_active = app.state.config_selector.config_active
    app.state.config = parse_yaml(BASE_PATH / "configs" / app.state.config_active)
    app.state.config_updates = copy.deepcopy(dict(app.state.config))
    app.state.model_active = app.state.config.detection.model.weights
    app.state.config_model = parse_json(BASE_PATH / "models" / app.state.config.detection.model.config)
    app.state.models = sorted([file.name for file in (BASE_PATH / "models").glob("*.blob")])
    app.state.configs = sorted([file.name for file in (BASE_PATH / "configs").glob("*.yaml")
                                if file.name != "config_selector.yaml"])

    # Initialize relevant app.state variables
    app.state.start_recording_after_shutdown = False
    app.state.show_overlay = False
    app.state.tracker_data = []
    app.state.labels = app.state.config_model.mappings.labels
    app.state.focus_initialized = False
    app.state.manual_focus_enabled = app.state.config.camera.focus.mode == "manual"
    app.state.focus_range_enabled = app.state.config.camera.focus.mode == "range"
    app.state.aspect_ratio = (app.state.config.webapp.resolution.width
                              / app.state.config.webapp.resolution.height)
    app.state.rec_durations = {
        "default": convert_duration(app.state.config.recording.duration.default),
        "battery": {level: convert_duration(getattr(app.state.config.recording.duration.battery, level))
                    for level in ["high", "medium", "low", "minimal"]}
    }
    app.state.fps = 0
    app.state.lens_pos = 0
    app.state.iso_sens = 0
    app.state.exp_time = 0
    app.state.frame_count = 0
    app.state.prev_time = time.monotonic()

    pipeline, app.state.sensor_res = create_pipeline(app.state.config, app.state.config_model)
    app.state.device = dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH)  # start device in USB2 mode

    # Create output queues to get the synchronized HQ frames and tracker + model output
    app.state.q_frame = app.state.device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    app.state.q_track = app.state.device.getOutputQueue(name="track", maxSize=4, blocking=False)

    # Create input queue to send control commands to OAK camera
    app.state.q_ctrl = app.state.device.getInputQueue(name="control", maxSize=4, blocking=False)

    ui.notification("OAK camera pipeline started!", type="positive", timeout=2)


async def setup_app():
    """Set up NiceGUI app and configure UI components."""
    await start_camera()

    # Create 1x1 black pixel PNG as placeholder image that will be shown when no frame is available
    placeholder_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    @app.get("/video/frame")
    async def serve_frame():
        """Serve MJPEG-encoded frame from OAK camera over HTTP and update camera parameters."""
        if hasattr(app.state, "q_frame") and app.state.q_frame and app.state.q_frame.has():
            frame_dai = app.state.q_frame.get()          # depthai.ImgFrame (type: BITSTREAM)
            frame_bytes = frame_dai.getData().tobytes()  # convert bitstream (numpy array) to bytes

            # Update FPS, lens position, ISO sensitivity and exposure time twice per second
            app.state.frame_count += 1
            current_time = time.monotonic()
            elapsed_time = current_time - app.state.prev_time
            if elapsed_time > 0.5:
                app.state.fps = round(app.state.frame_count / elapsed_time, 2)
                app.state.lens_pos = frame_dai.getLensPosition()
                app.state.iso_sens = frame_dai.getSensitivity()
                app.state.exp_time = frame_dai.getExposureTime().total_seconds()*1000  # milliseconds
                app.state.frame_count = 0
                app.state.prev_time = current_time

            return Response(content=frame_bytes, media_type="image/jpeg")
        else:
            return Response(content=placeholder_bytes, media_type="image/png")

    async def update_frame():
        """Update frame source with a timestamp to prevent caching."""
        app.state.frame_ii.set_source(f"/video/frame?{time.monotonic()}")

    async def update_tracker_data():
        """Update data from object tracker and detection model, set exposure region if enabled."""
        tracklets_data = []
        if hasattr(app.state, "q_track") and app.state.q_track and app.state.q_track.has():
            tracklets = app.state.q_track.get().tracklets
            for tracklet in tracklets:
                tracklet_status = tracklet.status.name
                if tracklet_status in {"TRACKED", "NEW"}:
                    tracklet_data = {
                        "label": app.state.labels[tracklet.srcImgDetection.label],
                        "confidence": round(tracklet.srcImgDetection.confidence, 2),
                        "track_ID": tracklet.id,
                        "track_status": tracklet_status,
                        "x_min": round(tracklet.srcImgDetection.xmin, 4),
                        "y_min": round(tracklet.srcImgDetection.ymin, 4),
                        "x_max": round(tracklet.srcImgDetection.xmax, 4),
                        "y_max": round(tracklet.srcImgDetection.ymax, 4)
                    }
                    tracklets_data.append(tracklet_data)
        app.state.tracker_data = tracklets_data

        if app.state.config_updates["detection"]["exposure_region"]["enabled"] and tracklets_data:
            # Use model bbox from most recent active tracklet to set auto exposure region
            tracklets_tracked = [t for t in tracklets_data if t["track_status"] == "TRACKED"]
            if tracklets_tracked:
                tracklet_max_id = max(tracklets_tracked, key=lambda t: t["track_ID"])
                roi_x, roi_y, roi_w, roi_h = convert_bbox_roi((tracklet_max_id["x_min"],
                                                               tracklet_max_id["y_min"],
                                                               tracklet_max_id["x_max"],
                                                               tracklet_max_id["y_max"]),
                                                               app.state.sensor_res)
                exp_ctrl = dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_w, roi_h)
                app.state.q_ctrl.send(exp_ctrl)

    async def update_overlay():
        """Update SVG overlay to show latest tracker/model data."""
        svg_overlay = [
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1" width="100%" height="100%" '
            'style="position:absolute; top:0; left:0; pointer-events:none;">'
        ]

        for data in app.state.tracker_data:
            label = data["label"]
            confidence = data["confidence"]
            track_id = data["track_ID"]
            x_min = (data["x_min"] - 0.5) * app.state.aspect_ratio + 0.5  # transform based on aspect ratio
            y_min = data["y_min"]
            x_max = (data["x_max"] - 0.5) * app.state.aspect_ratio + 0.5
            y_max = data["y_max"]
            width = x_max - x_min
            height = y_max - y_min

            # Add rectangle for bounding box
            svg_overlay.append(
                f'<rect x="{x_min}" y="{y_min}" width="{width}" height="{height}" '
                'fill="none" stroke="red" stroke-width="0.006" stroke-opacity="0.5" />'
            )

            # Add text for tracker/model data
            text_y = y_min + height + 0.04 if y_min + height < 0.95 else y_min - 0.05
            svg_overlay.append(
                f'<text x="{x_min}" y="{text_y}" '
                'font-size="0.04" fill="white" stroke="black" stroke-width="0.005" '
                'paint-order="stroke" text-anchor="start" font-weight="bold">'
                f'{label} {confidence}'
                f'<tspan x="{x_min}" dy="0.04">ID: {track_id}</tspan></text>'
            )

        svg_overlay.append("</svg>")
        app.state.frame_ii.set_content("".join(svg_overlay))

    async def update_frame_and_overlay():
        """Update frame and tracker/model data + overlay if enabled."""
        await update_frame()
        if app.state.show_overlay:
            await update_tracker_data()
            if app.state.tracker_data:
                await update_overlay()
            else:
                app.state.frame_ii.set_content("")
        else:
            app.state.frame_ii.set_content("")

    # Set timer to update frame (and overlay if enabled) depending on camera frame rate
    app.state.frame_timer = ui.timer(round(1 / app.state.config.webapp.fps, 3),
                                     update_frame_and_overlay)

    # Main content container (single column layout for responsive width and centering)
    with ui.column(align_items="center").classes("w-full max-w-3xl mx-auto"):

        @ui.refreshable
        def setup_ui_components():
            """Set up UI components and layout."""
            # Video stream container (responsive aspect ratio with padding technique)
            with ui.element("div").classes("w-full p-0 overflow-hidden bg-black border border-gray-700"):
                with ui.element("div").classes(f"relative w-full pb-[{100/app.state.aspect_ratio}%]"):
                    with ui.element("div").classes("absolute inset-0 flex items-center justify-center"):
                        app.state.frame_ii = (ui.interactive_image(content="")
                                              .classes("max-w-full max-h-full object-contain"))

            # Labels for current camera/frame parameters
            with ui.row(align_items="center").classes("w-full gap-2 -mt-3"):
                (ui.label().bind_text_from(app.state, "fps", lambda fps: f"FPS: {fps}")
                 .classes("font-bold text-xs"))
                ui.separator().props("vertical")
                (ui.label().bind_text_from(app.state, "lens_pos", lambda pos: f"Lens Position: {pos}")
                 .classes("font-bold text-xs"))
                ui.separator().props("vertical")
                (ui.label().bind_text_from(app.state, "iso_sens", lambda iso: f"ISO: {iso}")
                 .classes("font-bold text-xs"))
                ui.separator().props("vertical")
                (ui.label().bind_text_from(app.state, "exp_time", lambda exp: f"Exposure: {exp:.1f} ms")
                 .classes("font-bold text-xs"))

            # Switches to toggle dark mode and model/tracker overlay
            dark = ui.dark_mode()
            with ui.row(align_items="center").classes("w-full gap-4"):
                (ui.switch("Dark Mode", value=True).bind_value_to(dark)
                 .props("color=green").classes("font-bold"))
                ui.separator().props("vertical")
                (ui.switch("Model/Tracker Overlay").bind_value(app.state, "show_overlay")
                 .props("color=green").classes("font-bold"))

            # Slider for manual focus control (only visible if focus mode is set to "manual")
            async def set_manual_focus(e):
                """Set manual focus position of OAK camera."""
                if app.state.focus_initialized:
                    mf_ctrl = dai.CameraControl().setManualFocus(e.value)
                    app.state.q_ctrl.send(mf_ctrl)
                else:
                    app.state.focus_initialized = True

            with ui.column().classes("w-full gap-0 mb-0").bind_visibility_from(app.state, "manual_focus_enabled"):
                ui.label("Manual Focus:").classes("font-bold")
                (ui.slider(min=0, max=255, step=1, on_change=set_manual_focus)
                 .bind_value(app.state.config_updates["camera"]["focus"]["lens_position"], "manual")
                 .props("label"))

            # Slider for auto focus range control (only visible if focus mode is set to "range")
            async def preview_focus_range(e):
                """Set manual focus position of OAK camera to the last changed focus range position."""
                if app.state.focus_initialized and hasattr(app.state, "previous_lens_pos_range"):
                    if app.state.previous_lens_pos_range["min"] != e.value["min"]:
                        mf_ctrl = dai.CameraControl().setManualFocus(e.value["min"])
                        app.state.q_ctrl.send(mf_ctrl)
                    elif app.state.previous_lens_pos_range["max"] != e.value["max"]:
                        mf_ctrl = dai.CameraControl().setManualFocus(e.value["max"])
                        app.state.q_ctrl.send(mf_ctrl)
                else:
                    app.state.focus_initialized = True
                app.state.previous_lens_pos_range = e.value

            with ui.column().classes("w-full gap-0 mb-0").bind_visibility_from(app.state, "focus_range_enabled"):
                ui.label("Focus Range:").classes("font-bold")
                (ui.range(min=0, max=255, step=1, on_change=preview_focus_range)
                 .bind_value(app.state.config_updates["camera"]["focus"]["lens_position"], "range")
                 .props("label"))

            # Config file selector
            async def on_config_change(e):
                """Update active configuration file, reload configurations, and start new pipeline."""
                if e.value != app.state.config_active:
                    ui.notification("Restarting camera with new configuration!", type="info", timeout=2)
                    update_config_selector(e.value)
                    await restart_camera()

            with ui.row(align_items="center").classes("w-full gap-2 mt-0"):
                (ui.label("Active Config:").classes("font-bold whitespace-nowrap")
                 .tooltip("Activate config file that will be used by the web app and recording script"))
                (ui.select(app.state.configs, value=app.state.config_active, on_change=on_config_change)
                 .classes("flex-1 truncate"))

            # Card element containing all configuration settings
            with ui.card().props("flat bordered").classes("w-full"):

                def validate_number(value, min_value, max_value, multiple=None):
                    """Validate that number is within the required range."""
                    if multiple is None:
                        return value is not None and (min_value <= value <= max_value)
                    return value is not None and (min_value <= value <= max_value) and value % multiple == 0

                def grid_separator():
                    """Create a horizontal separator line for a 2-column grid layout."""
                    with ui.row().classes("w-full col-span-2 py-0 my-0"):
                        ui.element("div").classes("w-full border-t border-gray-700")

                # Camera settings
                async def on_focus_mode_change(e):
                    """Update relevant focus parameters in config, set continuous focus if selected."""
                    app.state.manual_focus_enabled = e.value == "manual"
                    app.state.focus_range_enabled = e.value == "range"
                    if e.value == "continuous":
                        af_ctrl = (dai.CameraControl()
                                   .setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO))
                        app.state.q_ctrl.send(af_ctrl)
                    else:
                        app.state.config_updates["camera"]["focus"]["distance"]["enabled"] = False
                        app.state.config_updates["camera"]["focus"]["lens_position"]["enabled"] = True

                with ui.expansion("Camera Settings", icon="photo_camera").classes("w-full font-bold"):
                    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

                        ui.label("Focus Mode").classes("font-bold")
                        (ui.select(["continuous", "manual", "range"], label="Focus",
                                   on_change=on_focus_mode_change)
                         .bind_value(app.state.config_updates["camera"]["focus"], "mode"))

                        grid_separator()
                        (ui.label("Frame Rate").classes("font-bold")
                         .tooltip("Higher FPS increases power consumption"))
                        (ui.number(label="FPS", placeholder=app.state.config.camera.fps,
                                   min=1, max=30, precision=0, step=1,
                                   validation={"Required value between 1-30":
                                               lambda v: validate_number(v, 1, 30)})
                         .bind_value(app.state.config_updates["camera"], "fps",
                                     forward=lambda v: int(v) if v is not None else None))

                        grid_separator()
                        (ui.label("Resolution").classes("font-bold")
                         .tooltip("Resolution of captured images (HQ frames)"))
                        with ui.row(align_items="center").classes("w-full gap-2"):
                            (ui.number(label="Width", placeholder=app.state.config.camera.resolution.width,
                                       min=320, max=3840, precision=0, step=32,
                                       validation={"Required value between 320-3840 (multiple of 32)":
                                                   lambda v: validate_number(v, 320, 3840, 32)})
                             .bind_value(app.state.config_updates["camera"]["resolution"], "width",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))
                            (ui.number(label="Height", placeholder=app.state.config.camera.resolution.height,
                                       min=320, max=2160, precision=0, step=2,
                                       validation={"Required value between 320-2160 (multiple of 2)":
                                                   lambda v: validate_number(v, 320, 2160, 2)})
                             .bind_value(app.state.config_updates["camera"]["resolution"], "height",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))

                        grid_separator()
                        (ui.label("JPEG Quality").classes("font-bold")
                         .tooltip("JPEG quality of captured images"))
                        (ui.number(label="JPEG", placeholder=app.state.config.camera.jpeg_quality,
                                   min=10, max=100, precision=0, step=1,
                                   validation={"Required value between 10-100":
                                               lambda v: validate_number(v, 10, 100)})
                         .bind_value(app.state.config_updates["camera"], "jpeg_quality",
                                     forward=lambda v: int(v) if v is not None else None))

                        grid_separator()
                        (ui.label("ISP Settings").classes("font-bold")
                         .tooltip("Setting Sharpness and Luma Denoise to 0 can reduce artifacts"))
                        with ui.row(align_items="center").classes("w-full gap-2"):
                            (ui.number(label="Sharpness",
                                       placeholder=app.state.config.camera.isp.sharpness,
                                       min=0, max=4, precision=0, step=1,
                                       validation={"Required value between 0-4":
                                                   lambda v: validate_number(v, 0, 4)})
                             .bind_value(app.state.config_updates["camera"]["isp"], "sharpness",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))
                            (ui.number(label="Luma Denoise",
                                       placeholder=app.state.config.camera.isp.luma_denoise,
                                       min=0, max=4, precision=1, step=1,
                                       validation={"Required value between 0-4":
                                                   lambda v: validate_number(v, 0, 4)})
                             .bind_value(app.state.config_updates["camera"]["isp"], "luma_denoise",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))
                            (ui.number(label="Chroma Denoise",
                                       placeholder=app.state.config.camera.isp.chroma_denoise,
                                       min=0, max=4, precision=1, step=1,
                                       validation={"Required value between 0-4":
                                                   lambda v: validate_number(v, 0, 4)})
                             .bind_value(app.state.config_updates["camera"]["isp"], "chroma_denoise",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))

                # Detection settings
                ui.separator()
                with ui.expansion("Detection Settings", icon="radar").classes("w-full font-bold"):
                    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

                        ui.label("Detection Model").classes("font-bold")
                        (ui.select(app.state.models, label="Model", value=app.state.model_active)
                         .bind_value(app.state.config_updates["detection"]["model"], "weights")
                         .bind_value_to(app.state.config_updates["detection"]["model"], "config",
                                        forward=lambda v: f"{Path(v).stem}.json" if v else None)
                         .classes("truncate"))

                        grid_separator()
                        (ui.label("Input Resolution").classes("font-bold")
                         .tooltip("Resolution of downscaled + stretched/cropped LQ frames for model input"))
                        with ui.row(align_items="center").classes("w-full gap-2"):
                            (ui.number(label="Width", placeholder=app.state.config.detection.resolution.width,
                                       min=128, max=640, precision=0, step=1,
                                       validation={"Required value between 128-640":
                                                   lambda v: validate_number(v, 128, 640)})
                             .bind_value(app.state.config_updates["detection"]["resolution"], "width",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))
                            (ui.number(label="Height", placeholder=app.state.config.detection.resolution.height,
                                       min=128, max=640, precision=0, step=1,
                                       validation={"Required value between 128-640":
                                                   lambda v: validate_number(v, 128, 640)})
                             .bind_value(app.state.config_updates["detection"]["resolution"], "height",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))

                        grid_separator()
                        (ui.label("Confidence Threshold").classes("font-bold")
                         .tooltip("Overrides model config file"))
                        (ui.number(label="Confidence", placeholder=app.state.config.detection.conf_threshold,
                                   min=0, max=1, precision=2, step=0.01,
                                   validation={"Required value between 0-1":
                                               lambda v: validate_number(v, 0, 1)})
                         .bind_value(app.state.config_updates["detection"], "conf_threshold"))

                        grid_separator()
                        (ui.label("IoU Threshold").classes("font-bold")
                         .tooltip("Overrides model config file"))
                        (ui.number(label="IoU", placeholder=app.state.config.detection.iou_threshold,
                                   min=0, max=1, precision=2, step=0.01,
                                   validation={"Required value between 0-1":
                                               lambda v: validate_number(v, 0, 1)})
                         .bind_value(app.state.config_updates["detection"], "iou_threshold"))

                        grid_separator()
                        (ui.label("Detection-based Exposure").classes("font-bold")
                         .tooltip("Use coordinates from most recent detection to set auto exposure region"))
                        (ui.switch("Enable")
                         .bind_value(app.state.config_updates["detection"]["exposure_region"], "enabled")
                         .props("color=green").classes("font-bold"))

                # Recording settings
                async def on_duration_change(e, duration_type, field_type):
                    """Convert hours and minutes to total minutes and update config."""
                    if e.value is not None:
                        if duration_type == "default":
                            duration_setting = app.state.rec_durations["default"]
                            config_target = app.state.config_updates["recording"]["duration"]
                        else:
                            duration_setting = app.state.rec_durations["battery"][duration_type]
                            config_target = app.state.config_updates["recording"]["duration"]["battery"]

                        duration_setting[field_type] = int(e.value)
                        hours = duration_setting.get("hours", 0) or 0  # default to 0 if not set
                        minutes = duration_setting.get("minutes", 0) or 0
                        total_minutes = (hours * 60) + minutes

                        duration_setting["total"] = total_minutes
                        if duration_type == "default":
                            config_target["default"] = total_minutes
                        else:
                            config_target[duration_type] = total_minutes

                def create_duration_inputs(duration_type, label_text, tooltip_text=None):
                    """Create hours and minutes input fields for a specific duration type."""
                    if duration_type == "default":
                        duration_setting = app.state.rec_durations["default"]
                    else:
                        duration_setting = app.state.rec_durations["battery"][duration_type]

                    label = ui.label(label_text).classes("font-bold")
                    if tooltip_text:
                        label.tooltip(tooltip_text)

                    with ui.row(align_items="center").classes("w-full gap-2"):
                        (ui.number(label="Hours", placeholder=1, min=0, max=24, precision=0, step=1,
                                   on_change=lambda e: on_duration_change(e, duration_type, "hours"),
                                   validation={"Required value between 0-24":
                                               lambda v: validate_number(v, 0, 24)})
                         .bind_value(duration_setting, "hours")).classes("flex-1")
                        (ui.number(label="Minutes", placeholder=0, min=0, max=59, precision=0, step=1,
                                   on_change=lambda e: on_duration_change(e, duration_type, "minutes"),
                                   validation={"Required value between 0-59":
                                               lambda v: validate_number(v, 0, 59)})
                         .bind_value(duration_setting, "minutes")).classes("flex-1")

                ui.separator()
                with ui.expansion("Recording Settings", icon="videocam").classes("w-full font-bold"):
                    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

                        (ui.label("Duration").classes("font-bold")
                         .tooltip("Duration per recording session"))
                        with ui.column().classes("w-full"):
                            with ui.tabs().classes("w-full") as tabs:
                                ui.tab("No Battery", icon="timer")
                                ui.tab("Battery", icon="battery_charging_full")

                            with ui.tab_panels(tabs, value="No Battery").classes("w-full"):
                                with ui.tab_panel("No Battery"):
                                    create_duration_inputs("default", "Default",
                                        "Duration if powermanager is disabled")
                                with ui.tab_panel("Battery"):
                                    create_duration_inputs("high", "High",
                                        "Duration if battery charge level is > 70% or USB power is connected")
                                    create_duration_inputs("medium", "Medium",
                                        "Duration if battery charge level is between 50-70%")
                                    create_duration_inputs("low", "Low",
                                        "Duration if battery charge level is between 30-50%")
                                    create_duration_inputs("minimal", "Minimal",
                                        "Duration if battery charge level is < 30%",)

                        grid_separator()
                        ui.label("Capture Interval").classes("font-bold")
                        with ui.column().classes("w-full"):
                            with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):
                                (ui.label("Detection").classes("font-bold")
                                 .tooltip("Interval for saving HQ frame + metadata while object is detected"))
                                (ui.number(label="Seconds",
                                           placeholder=app.state.config.recording.capture_interval.detection,
                                           min=0, max=60, precision=1, step=0.1,
                                           validation={"Required value between 0-60":
                                                       lambda v: validate_number(v, 0, 60)})
                                 .bind_value(app.state.config_updates["recording"]["capture_interval"], "detection"))
                                (ui.label("Timelapse").classes("font-bold")
                                 .tooltip("Interval for saving HQ frame (independent of detected objects)"))
                                (ui.number(label="Seconds",
                                           placeholder=app.state.config.recording.capture_interval.timelapse,
                                           min=0, max=3600, precision=1, step=0.1,
                                           validation={"Required value between 0-3600":
                                                       lambda v: validate_number(v, 0, 3600)})
                                 .bind_value(app.state.config_updates["recording"]["capture_interval"], "timelapse"))

                        grid_separator()
                        (ui.label("Shutdown After Recording").classes("font-bold")
                         .tooltip("Shut down Raspberry Pi after recording session is finished or interrupted"))
                        (ui.switch("Enable")
                         .bind_value(app.state.config_updates["recording"]["shutdown"], "enabled")
                         .props("color=green").classes("font-bold"))

                # Post-processing settings
                ui.separator()
                with ui.expansion("Post-Processing Settings", icon="tune").classes("w-full font-bold"):
                    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

                        (ui.label("Crop Detections").classes("font-bold")
                         .tooltip("Crop detections from HQ frames and save as individual .jpg images"))
                        with ui.column().classes("w-full gap-1"):
                            (ui.switch("Enable")
                             .bind_value(app.state.config_updates["post_processing"]["crop"], "enabled")
                             .props("color=green").classes("font-bold"))
                            (ui.select(["square", "original"], label="Crop Method")
                             .bind_visibility_from(app.state.config_updates["post_processing"]["crop"], "enabled")
                             .bind_value(app.state.config_updates["post_processing"]["crop"], "method")
                             .classes("w-full"))

                        grid_separator()
                        (ui.label("Draw Overlays").classes("font-bold")
                         .tooltip("Draw overlays on HQ frame copies (bounding box, label, confidence, track ID)"))
                        (ui.switch("Enable")
                         .bind_value(app.state.config_updates["post_processing"]["overlay"], "enabled")
                         .props("color=green").classes("font-bold"))

                        grid_separator()
                        (ui.label("Delete Originals").classes("font-bold")
                         .tooltip("Delete original HQ frames with detections after processing"))
                        (ui.switch("Enable")
                         .bind_value(app.state.config_updates["post_processing"]["delete"], "enabled")
                         .props("color=green").classes("font-bold"))

                        grid_separator()
                        (ui.label("Archive Data").classes("font-bold")
                         .tooltip("Archive (zip) all captured data + logs/configs and manage disk space"))
                        with ui.column().classes("w-full gap-1"):
                            (ui.switch("Enable")
                             .bind_value(app.state.config_updates["archive"], "enabled")
                             .props("color=green").classes("font-bold"))
                            (ui.number(label="Low Free Space", placeholder=app.state.config.archive.disk_low,
                                       min=100, max=50000, precision=0, step=100, suffix="MB",
                                       validation={"Required value between 100-50000 MB":
                                                   lambda v: validate_number(v, 100, 50000)})
                             .bind_visibility_from(app.state.config_updates["archive"], "enabled")
                             .bind_value(app.state.config_updates["archive"], "disk_low",
                                         forward=lambda v: int(v) if v is not None else None)
                             .tooltip("Minimum required free disk space for unarchived data retention")
                             .classes("w-full"))

                        grid_separator()
                        (ui.label("Upload to Cloud").classes("font-bold")
                         .tooltip("Upload archived data to cloud storage provider (always runs archive)"))
                        with ui.column().classes("w-full gap-1"):
                            (ui.switch("Enable")
                             .bind_value(app.state.config_updates["upload"], "enabled")
                             .props("color=green").classes("font-bold"))
                            (ui.select(["all", "full", "crop", "metadata"], label="Content")
                             .bind_visibility_from(app.state.config_updates["upload"], "enabled")
                             .bind_value(app.state.config_updates["upload"], "content")
                             .tooltip("Select content for upload, always including metadata")
                             .classes("w-full"))

                # System settings
                ui.separator()
                with ui.expansion("System Settings", icon="settings").classes("w-full font-bold"):
                    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

                        (ui.label("Power Management").classes("font-bold")
                         .tooltip("Disable if no power management board is connected"))
                        with ui.column().classes("w-full gap-1"):
                            (ui.switch("Enable")
                             .bind_value(app.state.config_updates["powermanager"], "enabled")
                             .props("color=green").classes("font-bold"))

                            with (ui.column().classes("w-full")
                                  .bind_visibility_from(app.state.config_updates["powermanager"], "enabled")):
                                (ui.select(["wittypi", "pijuice"], label="Board Model")
                                 .bind_value(app.state.config_updates["powermanager"], "model")
                                 .classes("w-full"))
                                with ui.row(align_items="center").classes("w-full gap-2"):
                                    (ui.number(label="Min. Charge",
                                               placeholder=app.state.config.powermanager.charge_min,
                                               min=10, max=90, precision=0, step=5, suffix="%",
                                               validation={"Required value between 10-90":
                                                           lambda v: validate_number(v, 10, 90)})
                                     .bind_value(app.state.config_updates["powermanager"], "charge_min",
                                                 forward=lambda v: int(v) if v is not None else None)
                                     .tooltip("Minimum required charge level to start/continue a recording")
                                     .classes("flex-1"))
                                    (ui.number(label="Check Interval",
                                               placeholder=app.state.config.powermanager.charge_check,
                                               min=5, max=300, precision=0, step=5, suffix="seconds",
                                               validation={"Required value between 5-300":
                                                           lambda v: validate_number(v, 5, 300)})
                                     .bind_value(app.state.config_updates["powermanager"], "charge_check",
                                                 forward=lambda v: int(v) if v is not None else None)
                                     .classes("flex-1"))

                        grid_separator()
                        (ui.label("OAK Temperature").classes("font-bold")
                         .tooltip("Maximum allowed OAK chip temperature to continue a recording"))
                        with ui.row(align_items="center").classes("w-full gap-2"):
                            (ui.number(label="Max. Temperature", placeholder=app.state.config.oak.temp_max,
                                       min=70, max=100, precision=0, step=1, suffix="°C",
                                       validation={"Required value between 70-100":
                                                   lambda v: validate_number(v, 70, 100)})
                             .bind_value(app.state.config_updates["oak"], "temp_max",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))
                            (ui.number(label="Check Interval", placeholder=app.state.config.oak.temp_check,
                                       min=5, max=300, precision=0, step=5, suffix="seconds",
                                       validation={"Required value between 5-300":
                                                   lambda v: validate_number(v, 5, 300)})
                             .bind_value(app.state.config_updates["oak"], "temp_check",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))

                        grid_separator()
                        ui.label("Storage Management").classes("font-bold")
                        with ui.row(align_items="center").classes("w-full gap-2"):
                            (ui.number(label="Min. Free Space", placeholder=app.state.config.storage.disk_min,
                                       min=100, max=10000, precision=0, step=100, suffix="MB",
                                       validation={"Required value between 100-10000 MB":
                                                   lambda v: validate_number(v, 100, 10000)})
                             .bind_value(app.state.config_updates["storage"], "disk_min",
                                         forward=lambda v: int(v) if v is not None else None)
                             .tooltip("Minimum required free disk space to start/continue a recording")
                             .classes("flex-1"))
                            (ui.number(label="Check Interval", placeholder=app.state.config.storage.disk_check,
                                       min=5, max=300, precision=0, step=5, suffix="seconds",
                                       validation={"Required value between 5-300":
                                                   lambda v: validate_number(v, 5, 300)})
                             .bind_value(app.state.config_updates["storage"], "disk_check",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))

                        grid_separator()
                        (ui.label("System Logging").classes("font-bold")
                         .tooltip("Log system information (temperature, memory, CPU utilization, battery info)"))
                        with ui.column().classes("w-full gap-1"):
                            (ui.switch("Enable")
                             .bind_value(app.state.config_updates["logging"], "enabled")
                             .props("color=green").classes("font-bold"))
                            (ui.number(label="Log Interval", placeholder=app.state.config.logging.interval,
                                       min=1, max=600, precision=0, step=1, suffix="seconds",
                                       validation={"Required value between 1-600":
                                                   lambda v: validate_number(v, 1, 600)})
                             .bind_visibility_from(app.state.config_updates["logging"], "enabled")
                             .bind_value(app.state.config_updates["logging"], "interval",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("w-full"))

                # Web app settings
                ui.separator()
                with ui.expansion("Web App Settings", icon="video_settings").classes("w-full font-bold"):
                    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

                        (ui.label("Frame Rate").classes("font-bold")
                         .tooltip("Max. possible streamed FPS depends on resolution"))
                        (ui.number(label="FPS", placeholder=app.state.config.webapp.fps,
                                   min=1, max=30, precision=0, step=1,
                                   validation={"Required value between 1-30":
                                               lambda v: validate_number(v, 1, 30)})
                         .bind_value(app.state.config_updates["webapp"], "fps",
                                     forward=lambda v: int(v) if v is not None else None))

                        grid_separator()
                        (ui.label("Resolution").classes("font-bold")
                         .tooltip("Resolution of streamed HQ frames"))
                        with ui.row(align_items="center").classes("w-full gap-2"):
                            (ui.number(label="Width", placeholder=app.state.config.webapp.resolution.width,
                                       min=320, max=1920, precision=0, step=32,
                                       validation={"Required value between 320-1920 (multiple of 32)":
                                                   lambda v: validate_number(v, 320, 1920, 32)})
                             .bind_value(app.state.config_updates["webapp"]["resolution"], "width",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))
                            (ui.number(label="Height", placeholder=app.state.config.webapp.resolution.height,
                                       min=320, max=1080, precision=0, step=2,
                                       validation={"Required value between 320-1080 (multiple of 2)":
                                                   lambda v: validate_number(v, 320, 1080, 2)})
                             .bind_value(app.state.config_updates["webapp"]["resolution"], "height",
                                         forward=lambda v: int(v) if v is not None else None)
                             .classes("flex-1"))

                        grid_separator()
                        (ui.label("JPEG Quality").classes("font-bold")
                         .tooltip("JPEG quality of streamed HQ frames"))
                        (ui.number(label="JPEG", placeholder=app.state.config.webapp.jpeg_quality,
                                   min=10, max=100, precision=0, step=1,
                                   validation={"Required value between 10-100":
                                               lambda v: validate_number(v, 10, 100)})
                         .bind_value(app.state.config_updates["webapp"], "jpeg_quality",
                                     forward=lambda v: int(v) if v is not None else None))


        # Set up all UI components and their bindings with option to refresh
        setup_ui_components()


        async def save_config():
            """Save configuration while preserving comments and structure."""
            if app.state.config_active == "config_default.yaml":
                ui.notification("Cannot save changes to default configuration!", type="warning", timeout=2)
                await create_new_config()
                return

            current_file_path = BASE_PATH / "configs" / app.state.config_active

            with ui.dialog() as dialog, ui.card():
                ui.label(f"Save changes to '{app.state.config_active}'?")
                with ui.row().classes("w-full justify-center gap-4 mt-4"):
                    ui.button("Cancel", on_click=lambda: dialog.submit("cancel"))
                    ui.button("Create New", on_click=lambda: dialog.submit("new"), color="green")
                    ui.button("Overwrite", on_click=lambda: dialog.submit("overwrite"), color="orange")

            result = await dialog
            if result == "cancel":
                ui.notification("Changes not saved!", type="warning", timeout=2)
            if result == "new":
                await create_new_config()
            elif result == "overwrite":
                await save_to_file(current_file_path)


        async def create_new_config():
            """Create a new configuration file."""

            async def show_name_input_dialog():
                """Show dialog to enter a name for the new configuration file."""
                with ui.dialog() as dialog, ui.card():
                    ui.label("Name for new config file:")
                    name_input = (ui.input(placeholder="config_custom",
                                           validation={"Please enter a valid filename":
                                                       lambda v: v is not None and all(c.isalnum() or c in
                                                                                       "_-" for c in v)})
                             .props("clearable autofocus suffix='.yaml'"))

                    with ui.row().classes("w-full justify-center gap-4 mt-4"):
                        ui.button("Cancel", on_click=lambda: dialog.submit("cancel"))
                        ui.button("Save", on_click=lambda: dialog.submit(name_input.value), color="green")

                return await dialog

            filename = await show_name_input_dialog()

            if not filename:
                ui.notification("Please enter a valid filename!", type="warning", timeout=2)
                return

            if filename == "cancel":
                ui.notification("New config creation cancelled!", type="warning", timeout=2)
                return

            config_new_path = BASE_PATH / "configs" / f"{filename}.yaml"
            if config_new_path.exists():
                if filename == "config_default.yaml":
                    ui.notification("Cannot overwrite default configuration!", type="warning", timeout=2)
                    return

                if filename == "config_selector.yaml":
                    ui.notification("Cannot overwrite config selector!", type="warning", timeout=2)
                    return

                with ui.dialog() as dialog, ui.card():
                    ui.label(f"File '{filename}.yaml' already exists.")
                    ui.label("What would you like to do?")
                    with ui.row().classes("w-full justify-center gap-4 mt-4"):
                        ui.button("Cancel", on_click=lambda: dialog.submit("cancel"))
                        ui.button("Enter New Name", on_click=lambda: dialog.submit("new_name"), color="green")
                        ui.button("Overwrite", on_click=lambda: dialog.submit("overwrite"), color="orange")

                result = await dialog
                if result == "cancel":
                    ui.notification("New config creation cancelled!", type="warning", timeout=2)
                    return

                if result == "new_name":
                    await create_new_config()
                    return

            await save_to_file(config_new_path)


        async def save_to_file(config_path):
            """Save configuration to specified file path."""
            config_default_path = BASE_PATH / "configs" / "config_default.yaml"

            with open(config_default_path, "r", encoding="utf-8") as file:
                config_text = file.read()

            config_text = config_text.replace(
                "# Insect Detect - Default Configuration Settings",
                "# Insect Detect - Custom Configuration Settings"
            ).replace(
                "# DO NOT MODIFY THIS DEFAULT CONFIG FILE - use \"config_custom.yaml\" for modifications",
                "# Use this custom config file for modifications (copy and create multiple configurations if needed)"
            )

            yaml = ruamel.yaml.YAML()
            yaml.width = 150  # maximum line width before wrapping
            yaml.preserve_quotes = True  # preserve all comments
            yaml.boolean_representation = ["false", "true"]  # ensure lowercase representation

            config_template = yaml.load(config_text)
            update_nested_dict(config_template, app.state.config_updates, dict(app.state.config))

            with open(config_path, "w", encoding="utf-8") as file:
                yaml.dump(config_template, file)

            ui.notification(f"Configuration saved to '{config_path.name}'!", type="positive", timeout=2)

            app.state.configs = sorted([file.name for file in (BASE_PATH / "configs").glob("*.yaml")
                                        if file.name != "config_selector.yaml"])

            if config_path.name == app.state.config_active:
                await show_apply_dialog(config_path.name)
            else:
                await show_activate_dialog(config_path.name)


        async def show_apply_dialog(config_name):
            """Show dialog to apply changes to current config."""
            with ui.dialog() as dialog, ui.card():
                ui.label(f"Configuration '{config_name}' has been updated.")
                ui.label("Do you want to apply the changes now?")

                with ui.row().classes("w-full justify-center gap-4 mt-4"):
                    ui.button("No", on_click=lambda: dialog.submit(False), color="red")
                    ui.button("Apply Changes", on_click=lambda: dialog.submit(True), color="green")

            apply_changes = await dialog
            if apply_changes:
                await apply_config_changes(config_name)
            else:
                ui.notification("Changes not applied!", type="warning", timeout=2)


        async def show_activate_dialog(config_name):
            """Show dialog to activate another config."""
            with ui.dialog() as dialog, ui.card():
                ui.label(f"Configuration saved to '{config_name}'")
                ui.label("Do you want to activate this configuration now?")

                with ui.row().classes("w-full justify-center gap-4 mt-4"):
                    ui.button("No", on_click=lambda: dialog.submit(False), color="red")
                    ui.button("Activate Config", on_click=lambda: dialog.submit(True), color="green")

            activate_config = await dialog
            if activate_config:
                await apply_config_changes(config_name)
            else:
                ui.notification("Configuration not activated!", type="warning", timeout=2)


        async def apply_config_changes(config_name):
            """Update config selector, set as active config and restart the camera."""
            ui.notification(f"Activating configuration '{config_name}'...", type="info", timeout=2)
            update_config_selector(config_name)
            app.state.config_active = config_name
            await restart_camera()


        async def restart_camera():
            """Disconnect from OAK device and restart camera with reloaded configurations."""
            n = ui.notification("Restarting camera pipeline...", type="ongoing", spinner=True, timeout=0)

            if hasattr(app.state, "frame_timer") and app.state.frame_timer is not None:
                app.state.frame_timer.deactivate()
                app.state.frame_timer = None

            if hasattr(app.state, "device") and app.state.device is not None:
                app.state.q_frame = None
                app.state.q_track = None
                app.state.q_ctrl = None
                app.state.device.close()
                app.state.device = None

            await asyncio.sleep(0.5)
            n.dismiss()
            await start_camera()

            setup_ui_components.refresh()
            app.state.frame_timer = ui.timer(round(1 / app.state.config.webapp.fps, 3),
                                             update_frame_and_overlay)


        async def start_recording():
            """Launch the recording script after shutting down the web app."""
            with ui.dialog() as dialog, ui.card():
                ui.label("Are you sure you want to stop the web app and start the recording script?")
                with ui.row().classes("w-full justify-center gap-4 mt-4"):
                    ui.button("Cancel", on_click=lambda: dialog.submit(False))
                    ui.button("Start Recording", on_click=lambda: dialog.submit(True),
                              color="teal", icon="play_circle")

            start = await dialog
            if start:
                app.state.start_recording_after_shutdown = True
                ui.notification("Stopping web app and start recording...", type="ongoing",
                                spinner=True, timeout=3)
                await asyncio.sleep(0.5)
                app.shutdown()


        async def confirm_shutdown():
            """Confirm or cancel shutdown of the web app."""
            with ui.dialog() as dialog, ui.card():
                ui.label("Are you sure you want to stop the web app?")
                with ui.row().classes("w-full justify-center gap-4 mt-4"):
                    ui.button("Cancel", on_click=lambda: dialog.submit(False))
                    ui.button("Stop App", on_click=lambda: dialog.submit(True),
                              color="red", icon="power_settings_new")

            shutdown = await dialog
            if shutdown:
                ui.notification("Stopping web app...", type="ongoing", spinner=True, timeout=3)
                await asyncio.sleep(0.5)
                app.shutdown()


        # Buttons to save config, start recording and stop web app
        with ui.row().classes("w-full justify-end mt-2 mb-4"):
            ui.button("Save Config", on_click=save_config, color="green", icon="save")
            ui.button("Start Recording", on_click=start_recording, color="teal", icon="play_circle")
            ui.button("Stop App", on_click=confirm_shutdown, color="red", icon="power_settings_new")

    # Print additional welcome message
    print(f"\nUse your hostname to open the Insect Detect web app: http://{CAM_ID}:5000")


    async def disconnect():
        """Disconnect all clients from currently running server."""
        for client_id in Client.instances:
            await core.sio.disconnect(client_id)


    async def cleanup():
        """Disconnect clients and close running OAK device, start recording if requested."""
        await disconnect()

        if hasattr(app.state, "device") and app.state.device is not None:
            app.state.device.close()

        if hasattr(app.state, "start_recording_after_shutdown") and app.state.start_recording_after_shutdown:
            log_file = BASE_PATH / "subprocess_log.log"
            timestamp = datetime.now().strftime("%F %T")
            log_entry = f"{timestamp} - Running yolo_tracker_save_hqsync.py\n"

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)

            with open(log_file, "a", encoding="utf-8") as log_file_handle:
                subprocess.Popen(
                    [sys.executable, str(BASE_PATH / "yolo_tracker_save_hqsync.py")],
                    stdout=log_file_handle,
                    stderr=log_file_handle,
                    start_new_session=True
                )

        # Force exit web app after timeout
        loop = asyncio.get_event_loop()
        loop.call_later(10, lambda: print("\nWeb app forced to exit after timeout.") or sys.exit(0))


    def signal_handler(signum, frame):
        """Handle a received signal (e.g. keyboard interrupt) to gracefully shut down the app."""
        print("\nSignal received, initiating graceful app shutdown...")
        app.shutdown()


    # Register signal handler for graceful shutdown if SIGINT or SIGTERM is received
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Register cleanup function to be called on app shutdown
    app.on_shutdown(cleanup)

# Set up and run the app
app.on_startup(setup_app)
ui.run(host="0.0.0.0", port=5000, title=f"{CAM_ID} Web App",
       favicon=str(BASE_PATH / "static" / "favicon.ico"),
       show=False, reload=False)
