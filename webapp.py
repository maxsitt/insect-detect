"""Stream OAK camera live feed and configure settings via NiceGUI-based web app.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Run this script with the Python interpreter from the virtual environment where you installed
the required packages, e.g. with 'env_insdet/bin/python3 insect-detect/webapp.py'.

Optional arguments:
'-config' set path to YAML file that contains all configuration parameters
          -> e.g. '-config configs/config_custom.yaml' to use custom config file

- load YAML file with configuration parameters and JSON file with detection model parameters
- stream frames (MJPEG-encoded bitstream) from OAK camera to browser-based web app
- draw tracker/model output overlay on frames using client-side SVG
- control camera settings via web app
- save modified configuration parameters to new config file

partly based on scripts from https://github.com/luxonis and https://github.com/zauberzeug/nicegui
"""

import argparse
import base64
import copy
import signal
import socket
import time
from datetime import timedelta
from pathlib import Path

import depthai as dai
import ruamel.yaml
from fastapi import Response
from nicegui import Client, app, core, ui

from utils.config import parse_json, parse_yaml
from utils.oak import convert_cm_lens_position

# Define optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("-config", type=str, default="configs/config_default.yaml",
    help="Set path to YAML file with configuration parameters.")
args = parser.parse_args()

# Set camera trap ID (default: hostname)
CAM_ID = socket.gethostname()

# Set base path (default: "insect-detect" directory)
BASE_PATH = Path.home() / "insect-detect"

# Parse configuration files
config = parse_yaml(BASE_PATH / args.config)
config_model = parse_json(BASE_PATH / config.detection.model.config)
config_updates = copy.deepcopy(dict(config))

# Extract some relevant configuration parameters
FPS_WEBAPP = config.webapp.fps
RES_HQ_STREAM = (config.webapp.resolution.width, config.webapp.resolution.height)
aspect_ratio_hq = RES_HQ_STREAM[0] / RES_HQ_STREAM[1]
RES_LQ = (config.detection.resolution.width, config.detection.resolution.height)

# Get all currently available detection models (.blob format)
blob_files = sorted([file.name for file in (BASE_PATH / "models").glob("*.blob")])
configured_model = Path(config.detection.model.weights).name

# Create black placeholder image that will be shown if no frame is available
black_1px = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII="
placeholder = Response(content=base64.b64decode(black_1px.encode("ascii")), media_type="image/png")


def create_pipeline():
    """Create and configure depthai pipeline for OAK camera."""
    pipeline = dai.Pipeline()

    # Create and configure color camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setFps(FPS_WEBAPP)  # frames per second available for focus/exposure and model input
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    SENSOR_RES = cam_rgb.getResolutionSize()
    cam_rgb.setInterleaved(False)  # planar layout
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    if RES_HQ_STREAM[0] > 1280 or RES_HQ_STREAM[1] > 720:
        cam_rgb.setIspScale(1, 2)  # use ISP to downscale 4K to 1080p resolution -> HQ stream
        if RES_HQ_STREAM[0] < 1920 or RES_HQ_STREAM[1] < 1080:
            cam_rgb.setVideoSize(*RES_HQ_STREAM)  # crop to configured HQ resolution
        else:
            cam_rgb.setVideoSize(1920, 1080)      # always cap resolution at 1080p for HQ stream
    else:
        cam_rgb.setIspScale(1, 3)  # use ISP to downscale 4K to 720p resolution -> HQ stream
        if RES_HQ_STREAM != (1280, 720):
            cam_rgb.setVideoSize(*RES_HQ_STREAM)

    cam_rgb.setPreviewSize(*RES_LQ)               # downscale frames for model input -> LQ frames
    if abs(aspect_ratio_hq - 1) > 0.01:           # check if HQ resolution is not ~1:1 aspect ratio
        cam_rgb.setPreviewKeepAspectRatio(False)  # stretch LQ frames to square for model input

    if config.camera.focus.mode == "range":
        if config.camera.focus.distance.enabled:
            lens_pos_min = convert_cm_lens_position(config.camera.focus.distance.range.max)
            lens_pos_max = convert_cm_lens_position(config.camera.focus.distance.range.min)
            cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)
        elif config.camera.focus.lens_position.enabled:
            lens_pos_min = config.camera.focus.lens_position.range.min
            lens_pos_max = config.camera.focus.lens_position.range.max
            cam_rgb.initialControl.setAutoFocusLensRange(lens_pos_min, lens_pos_max)
    elif config.camera.focus.mode == "manual":
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
    cam_rgb.video.link(encoder.input)

    # Create and configure YOLO detection network node and define input
    yolo = pipeline.create(dai.node.YoloDetectionNetwork)
    yolo.setBlobPath(BASE_PATH / config.detection.model.weights)
    yolo.setConfidenceThreshold(config.detection.conf_threshold)
    yolo.setIouThreshold(config.detection.iou_threshold)
    yolo.setNumClasses(config_model.nn_config.NN_specific_metadata.classes)
    yolo.setCoordinateSize(config_model.nn_config.NN_specific_metadata.coordinates)
    yolo.setAnchors(config_model.nn_config.NN_specific_metadata.anchors)
    yolo.setAnchorMasks(config_model.nn_config.NN_specific_metadata.anchor_masks)
    yolo.setNumInferenceThreads(2)
    cam_rgb.preview.link(yolo.input)
    yolo.input.setBlocking(False)

    # Create and configure object tracker node and define inputs
    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    yolo.passthrough.link(tracker.inputTrackerFrame)
    yolo.passthrough.link(tracker.inputDetectionFrame)
    yolo.out.link(tracker.inputDetections)

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

    return pipeline


def setup_app():
    """Start OAK camera pipeline and set up NiceGUI app."""

    # Connect to OAK device and start pipeline in USB2 mode
    pipeline = create_pipeline()
    app.state.device = dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH)

    # Create output queues to get the synchronized HQ frames and tracker + model output
    app.state.q_frame = app.state.device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    app.state.q_track = app.state.device.getOutputQueue(name="track", maxSize=4, blocking=False)

    # Create input queue to send control commands to OAK camera
    app.state.q_ctrl = app.state.device.getInputQueue(name="control", maxSize=4, blocking=False)

    # Set relevant app.state variables
    app.state.show_overlay = False
    app.state.tracker_data = []
    app.state.labels = config_model.mappings.labels
    app.state.focus_initialized = False
    app.state.manual_focus_enabled = config.camera.focus.mode == "manual"
    app.state.focus_range_enabled = config.camera.focus.mode == "range"
    app.state.lens_pos_manual = config.camera.focus.lens_position.manual
    app.state.lens_pos_range = {"min": config.camera.focus.lens_position.range.min,
                                "max": config.camera.focus.lens_position.range.max}
    app.state.fps = 0
    app.state.frame_count = 0
    app.state.prev_time = time.monotonic()

    @app.get("/video/frame")
    async def fetch_frame():
        """Return the latest MJPEG-encoded frame from OAK camera."""
        if app.state.q_frame.has():
            frame_dai = app.state.q_frame.get()  # depthai.ImgFrame
            frame = frame_dai.getData()          # numpy.ndarray
            app.state.lens_pos = frame_dai.getLensPosition()

            # Update FPS counter
            app.state.frame_count += 1
            current_time = time.monotonic()
            elapsed_time = current_time - app.state.prev_time

            # Calculate fps every second
            if elapsed_time > 1:
                app.state.fps = round(app.state.frame_count / elapsed_time, 2)
                app.state.frame_count = 0
                app.state.prev_time = current_time

            return Response(content=frame.tobytes(), media_type="image/jpeg")
        else:
            return placeholder

    def update_frame():
        """Update frame source with a timestamp to prevent caching."""
        app.state.frame_ii.set_source(f"/video/frame?{time.time()}")

    def fetch_tracker_data():
        """Fetch tracker/model output."""
        if app.state.q_track.has():
            tracklets = app.state.q_track.get().tracklets
            tracklets_data = []
            for tracklet in tracklets:
                if tracklet.status.name in {"TRACKED", "NEW"}:
                    tracklet_data = {
                        "label": app.state.labels[tracklet.srcImgDetection.label],
                        "confidence": round(tracklet.srcImgDetection.confidence, 2),
                        "track_ID": tracklet.id,
                        "x_min": round(tracklet.srcImgDetection.xmin, 4),
                        "y_min": round(tracklet.srcImgDetection.ymin, 4),
                        "x_max": round(tracklet.srcImgDetection.xmax, 4),
                        "y_max": round(tracklet.srcImgDetection.ymax, 4),
                    }
                    tracklets_data.append(tracklet_data)
            app.state.tracker_data = tracklets_data

    def update_overlay():
        """Update the SVG overlay based on current tracker/model output."""
        if not app.state.show_overlay:
            return

        svg_overlay = [
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1" width="100%" height="100%" '
            'style="position:absolute; top:0; left:0; pointer-events:none;">'
        ]

        for data in app.state.tracker_data:
            label = data["label"]
            confidence = data["confidence"]
            track_id = data["track_ID"]
            x_min = (data["x_min"] - 0.5) * aspect_ratio_hq + 0.5  # transform based on aspect ratio
            y_min = data["y_min"]
            x_max = (data["x_max"] - 0.5) * aspect_ratio_hq + 0.5
            y_max = data["y_max"]
            width = x_max - x_min
            height = y_max - y_min

            # Add rectangle for bounding box
            svg_overlay.append(
                f'<rect x="{x_min}" y="{y_min}" width="{width}" height="{height}" '
                'fill="none" stroke="red" stroke-width="0.006" stroke-opacity="0.5" />'
            )

            # Add text for tracker/model output
            text_y = y_min + height + 0.04 if y_min + height < 0.95 else y_min - 0.05
            svg_overlay.append(
                f'<text x="{x_min}" y="{text_y}" '
                'font-size="0.04" fill="white" stroke="black" stroke-width="0.005" '
                'paint-order="stroke" text-anchor="start" font-weight="bold">'
                f'{label} {confidence}'
                f'<tspan x="{x_min}" dy="0.04">ID: {track_id}</tspan></text>'
            )

        svg_overlay.append("</svg>")
        app.state.frame_ii.content = "".join(svg_overlay)

    def update_frame_and_overlay():
        """Update frame and tracker/model output + overlay if enabled."""
        update_frame()
        if app.state.show_overlay:
            fetch_tracker_data()
            update_overlay()

    # Create UI components
    with ui.column().classes("w-full max-w-3xl mx-auto items-center"):

        # Video stream container
        with ui.element("div").classes("w-full p-0 overflow-hidden bg-black border border-gray-700"):
            with ui.element("div").classes(f"relative w-full pb-[{100/aspect_ratio_hq}%]"):
                with ui.element("div").classes("absolute inset-0 flex items-center justify-center"):
                    app.state.frame_ii = ui.interactive_image(content="").classes("max-w-full max-h-full object-contain")

        # FPS label, lens position label and overlay toggle switch
        with ui.row().classes("w-full justify-between items-center"):
            ui.label().bind_text_from(app.state, "fps", lambda fps: f"FPS: {fps}").classes("font-bold")
            ui.label().bind_text_from(app.state, "lens_pos", lambda pos: f"Lens Position: {pos}").classes("font-bold")

            def toggle_overlay(e):
                """Toggle the tracker/model overlay visibility."""
                app.state.show_overlay = e.value
                if not app.state.show_overlay:
                    app.state.frame_ii.content = ""

            ui.switch("Model/Tracker Overlay", value=app.state.show_overlay,
                      on_change=toggle_overlay).props("color=green").classes("font-bold")

        # Slider for manual focus control
        with ui.column().classes("w-full gap-0"):

            def set_manual_focus():
                """Set manual focus position of OAK camera."""
                if app.state.focus_initialized:
                    mf_ctrl = dai.CameraControl().setManualFocus(app.state.lens_pos_manual)
                    app.state.q_ctrl.send(mf_ctrl)
                    config_updates["camera"]["focus"]["lens_position"].update({"manual": app.state.lens_pos_manual})
                else:
                    app.state.focus_initialized = True

            ui.label("Manual Focus:").bind_visibility_from(app.state, "manual_focus_enabled").classes("font-bold")
            (ui.slider(min=0, max=255, step=1, on_change=set_manual_focus)
             .bind_visibility_from(app.state, "manual_focus_enabled")
             .bind_value(app.state, "lens_pos_manual").props("label"))

        # Slider for auto focus range control
        with ui.column().classes("w-full gap-0"):

            def preview_focus_range():
                """Set manual focus position of OAK camera based on which focus range position is updated."""
                if not hasattr(app.state, "previous_lens_pos_range"):
                    app.state.previous_lens_pos_range = app.state.lens_pos_range
                    app.state.focus_initialized = True
                    return

                if app.state.previous_lens_pos_range["min"] != app.state.lens_pos_range["min"]:
                    mf_ctrl = dai.CameraControl().setManualFocus(app.state.lens_pos_range["min"])
                    app.state.q_ctrl.send(mf_ctrl)
                elif app.state.previous_lens_pos_range["max"] != app.state.lens_pos_range["max"]:
                    mf_ctrl = dai.CameraControl().setManualFocus(app.state.lens_pos_range["max"])
                    app.state.q_ctrl.send(mf_ctrl)
                config_updates["camera"]["focus"]["lens_position"].update({"range": app.state.lens_pos_range})
                app.state.previous_lens_pos_range = app.state.lens_pos_range

            ui.label("Focus Range:").bind_visibility_from(app.state, "focus_range_enabled").classes("font-bold")
            (ui.range(min=0, max=255, step=1, on_change=preview_focus_range)
             .bind_visibility_from(app.state, "focus_range_enabled")
             .bind_value(app.state, "lens_pos_range").props("label"))

        # Configuration settings
        with ui.card().classes("w-full"):
            ui.label("Configuration Parameters").classes("text-xl font-bold")

            # Camera Section
            with ui.expansion("Camera Settings", icon="photo_camera").classes("w-full"):
                with ui.grid(columns=2).classes("w-full gap-4 items-center"):

                    ui.label("Frame Rate (FPS)")
                    with ui.element("div").classes("w-full flex items-center"):
                        ui.number(min=1, max=30, step=1, value=config.camera.fps,
                                  on_change=lambda e: config_updates["camera"].update({"fps": int(e.value)})).classes("w-full")

                    ui.label("Focus Mode")
                    with ui.element("div").classes("w-full flex items-center"):

                        def handle_focus_mode_change(e):
                            """Handle focus mode change event."""
                            config_updates["camera"]["focus"].update({"mode": e.value})
                            app.state.manual_focus_enabled = e.value == "manual"
                            app.state.focus_range_enabled = e.value == "range"
                            if e.value == "continuous":
                                af_ctrl = dai.CameraControl().setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
                                app.state.q_ctrl.send(af_ctrl)
                            elif e.value == "manual":
                                config_updates["camera"]["focus"]["lens_position"].update({"enabled": True})
                                config_updates["camera"]["focus"]["distance"].update({"enabled": False})
                                config_updates["camera"]["focus"]["lens_position"].update({"manual": app.state.lens_pos_manual})
                            elif e.value == "range":
                                config_updates["camera"]["focus"]["lens_position"].update({"enabled": True})
                                config_updates["camera"]["focus"]["distance"].update({"enabled": False})
                                config_updates["camera"]["focus"]["lens_position"].update({"range": app.state.lens_pos_range})

                        ui.select(["continuous", "manual", "range"], value=config.camera.focus.mode,
                                  on_change=handle_focus_mode_change).classes("w-full")

            # Detection Section
            with ui.expansion("Detection Settings", icon="track_changes").classes("w-full"):
                with ui.grid(columns=2).classes("w-full gap-4 items-center"):

                    ui.label("Model Weights")
                    with ui.element("div").classes("w-full flex items-center"):

                        def update_model_paths(e):
                            """Update both model weights and config paths."""
                            blob_name = e.value
                            json_name = blob_name.replace(".blob", ".json")
                            config_updates["detection"]["model"].update({
                                "weights": f"models/{blob_name}",
                                "config": f"models/{json_name}"
                            })

                        ui.select(blob_files, value=configured_model,
                                  on_change=update_model_paths).classes("w-full")

                    ui.label("Confidence Threshold")
                    with ui.element("div").classes("w-full flex items-center"):
                        ui.number(value=config.detection.conf_threshold, min=0, max=1, step=0.01,
                                  on_change=lambda e: config_updates["detection"].update({"conf_threshold": e.value})).classes("w-full")

            # Save modified configuration
            with ui.row().classes("w-full justify-end mt-4"):

                async def save_config():
                    """Save modified configuration while preserving comments and structure."""

                    def update_nested_dict(template, updates):
                        """Update nested dictionary recursively while preserving key order."""
                        for key, value in updates.items():
                            if isinstance(value, dict) and isinstance(template.get(key), dict):
                                update_nested_dict(template[key], value)
                            else:
                                template[key] = value

                    config_default_path = BASE_PATH / "configs" / "config_default.yaml"
                    config_webapp_path = BASE_PATH / "configs" / "config_custom_webapp.yaml"

                    with open(config_default_path, "r", encoding="utf-8") as file:
                        config_text = file.read()

                    config_text = config_text.replace(
                        "# Insect Detect - Default Configuration Settings", 
                        "# Insect Detect - Custom Configuration Settings"
                    ).replace(
                        "# DO NOT MODIFY THIS DEFAULT CONFIG FILE - use \"config_custom.yaml\" for modifications",
                        "# This custom config file was generated by the Insect Detect web app"
                    )

                    yaml = ruamel.yaml.YAML()
                    yaml.width = 150
                    yaml.preserve_quotes = True  # preserve all comments
                    config_template = yaml.load(config_text)

                    update_nested_dict(config_template, config_updates)

                    with open(config_webapp_path, "w", encoding="utf-8") as file:
                        yaml.dump(config_template, file)

                    ui.notify("Configuration saved successfully", color="positive")

                ui.button("Save Configuration", on_click=save_config, color="green", icon="save")

        # Shutdown button
        with ui.row().classes("w-full justify-center"):

            async def confirm_shutdown():
                """Confirm or cancel shutdown of the web app."""
                with ui.dialog() as dialog, ui.card():
                    ui.label("Are you sure you want to shut down the web app?")
                    with ui.row().classes("w-full justify-center gap-4 mt-4"):
                        ui.button("Cancel", on_click=lambda: dialog.close())
                        ui.button("Shutdown", on_click=lambda: [dialog.close(), app.shutdown()], color="red", icon="power_settings_new")
                await dialog

            ui.button("Shutdown", on_click=confirm_shutdown, color="red", icon="power_settings_new").classes("font-bold")

    # Set timer to update frame (and overlay) depending on camera frame rate
    ui.timer(1/FPS_WEBAPP, update_frame_and_overlay)

    # Print additional welcome message
    print(f"\nYou can also use http://{CAM_ID}:5000 to open the Insect Detect web app.")

    async def disconnect():
        """Disconnect all clients from current running server."""
        for client_id in Client.instances:
            await core.sio.disconnect(client_id)

    async def cleanup():
        """Cleanup function to be called on application shutdown."""
        await disconnect()
        if hasattr(app.state, "device") and app.state.device is not None:
            app.state.device.close()

    def signal_handler(signum, frame):
        """Handle a received signal to gracefully shut down the application."""
        ui.timer(0.1, disconnect, once=True)
        ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)

    # Register signal handler for graceful shutdown
    app.on_shutdown(cleanup)
    signal.signal(signal.SIGINT, signal_handler)

# Set up and run the app
app.on_startup(setup_app)
ui.run(host="0.0.0.0", port=5000, title=f"{CAM_ID} Camera Control",
       favicon=str(BASE_PATH / "static" / "favicon.ico"),
       dark=True, show=False, reload=False)
