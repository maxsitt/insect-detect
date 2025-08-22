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
- stream MJPEG-encoded frames from OAK camera to browser-based web app via HTTP
- draw SVG overlay with model/tracker data on frames (bounding box, label, confidence, tracking ID)
- note relevant metadata for each deployment (e.g. start time, location, setting, field notes)
- configure camera, web app, recording and system settings via web app interface
- save modified configuration parameters to config file
- optionally start recording session with specified configuration parameters

partly based on open source scripts available at https://github.com/zauberzeug/nicegui
"""

import asyncio
import base64
import copy
import logging
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import depthai as dai
from fastapi.responses import StreamingResponse
from gpiozero import LED
from nicegui import Client, app, binding, core, run, ui

from utils.app import convert_duration, create_duration_inputs, grid_separator, validate_number
from utils.config import check_config_changes, parse_json, parse_yaml, update_config_file, update_config_selector
from utils.log import subprocess_log
from utils.network import get_current_connection, get_ip_address, set_up_network
from utils.oak import convert_bbox_roi, create_pipeline

# Set base path and get hostname + IP address
BASE_PATH = Path.home() / "insect-detect"
HOSTNAME = socket.gethostname()
IP_ADDRESS = get_ip_address()

# Create directory where logs will be stored
LOGS_PATH = BASE_PATH / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Set paths for marker files to indicate web app auto-run and streaming mode
AUTO_RUN_MARKER = BASE_PATH / ".auto_run_active"
STREAMING_MARKER = BASE_PATH / ".streaming_active"  # indicates user interaction with web app

# Create 1x1 black pixel PNG as placeholder image that will be shown when no frame is available
PLACEHOLDER_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
PLACEHOLDER_PNG_BYTES_LENGTH = str(len(PLACEHOLDER_PNG_BYTES)).encode()

# Set available GPIO pins (BCM numbering) for LED (excluding pins that are used by Witty Pi 4 L3V7)
LED_GPIO_PINS = [18, 23, 24, 25, 8, 7, 12, 16, 20, 21, 27, 22, 10, 9, 11, 13, 19, 26]

# Increase threshold for max. binding propagation time to avoid early warning messages
binding.MAX_PROPAGATION_TIME = 0.05  # default: 0.01 seconds

# Set fields that are allowed to be empty in the config file (won't be replaced with default value)
# For hotspot and Wi-Fi connections 'ssid' and 'password' are always allowed to be empty
OPTIONAL_CONFIG_FIELDS = {
    "deployment.start",
    "deployment.location.latitude",
    "deployment.location.longitude",
    "deployment.location.accuracy",
    "deployment.setting",
    "deployment.distance",
    "deployment.notes",
    "startup.auto_run.fallback"
}

# Set logging levels and format, stream logs to console and save to file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(f"{LOGS_PATH}/{Path(__file__).stem}.log",
                                                  encoding="utf-8")])
logger = logging.getLogger()
logger.info("-------- Web App Logger initialized --------")


@ui.page("/")
async def main_page():
    """Main entry point for the web app."""
    if AUTO_RUN_MARKER.exists() and not STREAMING_MARKER.exists():
        # Create marker file to indicate user interaction if in auto-run mode
        STREAMING_MARKER.touch()

    # Start camera if not already running
    if not getattr(app.state, "device", None):
        await start_camera()

    # Create main UI content container (single column layout for responsive width and centering)
    with ui.column(align_items="center").classes("w-full max-w-3xl mx-auto"):
        create_ui_layout()

    # Create timer to get latest model/tracker data and update overlay (capped to 10 Hz)
    app.state.overlay_timer = ui.timer(max(app.state.refresh_interval, 0.1),
                                       update_overlay, immediate=False)

    # Slow-blink LED to indicate web app is running and user is connected
    if getattr(app.state, "config", None) and app.state.config.led.enabled:
        led_gpio_pin = app.state.config.led.gpio_pin
        for _ in range(30):  # retry for 3 seconds as LED might still be used by other process
            try:
                app.state.led = LED(led_gpio_pin)
                break
            except Exception:
                await asyncio.sleep(0.1)
    if getattr(app.state, "led", None):
        app.state.led.blink(on_time=1, off_time=1, background=True)


@ui.refreshable
def create_ui_layout():
    """Define layout for all UI elements."""
    create_video_stream_container()
    create_control_elements()

    with ui.card().tight().classes("w-full"):
        with ui.expansion("Deployment", icon="location_on").classes("w-full font-bold"):
            create_deployment_section()

    with ui.card().tight().classes("w-full"):
        with ui.expansion("Configuration", icon="settings").classes("w-full font-bold"):
            with ui.expansion("Camera Settings", icon="photo_camera").classes("w-full font-bold"):
                create_camera_settings()
            ui.separator()
            with ui.expansion("Detection Settings", icon="radar").classes("w-full font-bold"):
                create_detection_settings()
            ui.separator()
            with ui.expansion("Recording Settings", icon="videocam").classes("w-full font-bold"):
                create_recording_settings()
            ui.separator()
            with ui.expansion("Post-Processing Settings", icon="tune").classes("w-full font-bold"):
                create_processing_settings()
            ui.separator()
            with ui.expansion("Startup Settings", icon="rocket_launch").classes("w-full font-bold"):
                create_startup_settings()
            ui.separator()
            with ui.expansion("Web App Settings", icon="video_settings").classes("w-full font-bold"):
                create_webapp_settings()
            ui.separator()
            with ui.expansion("System Settings", icon="settings_applications").classes("w-full font-bold"):
                create_system_settings()
            ui.separator()
            with ui.expansion("Network Settings", icon="network_wifi").classes("w-full font-bold"):
                create_network_settings()

    with ui.card().tight().classes("w-full"):
        with ui.expansion("Advanced", icon="build").classes("w-full font-bold"):
            with ui.expansion("View Logs", icon="article").classes("w-full font-bold"):
                create_logs_section()

    with ui.row().classes("w-full justify-end mt-2 mb-4 gap-2"):
        (ui.button("Save Conf", on_click=save_config, color="green", icon="save")
         .props("dense"))
        (ui.button("Start Rec", on_click=start_recording, color="teal", icon="play_circle")
         .props("dense"))
        (ui.button("Stop App", on_click=confirm_shutdown, color="red", icon="power_settings_new")
         .props("dense"))


@app.get("/video/stream")
async def stream_mjpeg():
    """Stream MJPEG-encoded frames from OAK camera over HTTP."""
    return StreamingResponse(content=frame_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


def create_video_stream_container():
    """Create video stream container with responsive aspect ratio and row with camera parameters."""
    with ui.element("div").classes("w-full p-0 overflow-hidden bg-black border border-gray-700"):
        with ui.element("div").classes(f"relative w-full pb-[{100/app.state.aspect_ratio}%]"):
            with ui.element("div").classes("absolute inset-0 flex items-center justify-center"):
                app.state.frame_ii = (ui.interactive_image(source="/video/stream")
                                      .classes("max-w-full max-h-full object-contain"))

    with ui.row(align_items="center").classes("w-full gap-2 -mt-3"):
        (ui.label().classes("font-bold text-xs")
         .bind_text_from(app.state, "fps", lambda fps: f"FPS: {fps}"))
        ui.separator().props("vertical")
        (ui.label().classes("font-bold text-xs")
         .bind_text_from(app.state, "lens_pos", lambda pos: f"Lens Position: {pos}"))
        ui.separator().props("vertical")
        (ui.label().classes("font-bold text-xs")
         .bind_text_from(app.state, "iso_sens", lambda iso: f"ISO: {iso}"))
        ui.separator().props("vertical")
        (ui.label().classes("font-bold text-xs")
         .bind_text_from(app.state, "exp_time", lambda exp: f"Exposure: {exp:.1f} ms"))


async def start_camera():
    """Connect to OAK device and start camera with selected configuration."""

    # Parse active config file and load configuration parameters
    app.state.config_selector = parse_yaml(BASE_PATH / "configs" / "config_selector.yaml")
    app.state.config_active = app.state.config_selector.config_active
    app.state.config = parse_yaml(BASE_PATH / "configs" / app.state.config_active)
    app.state.config_updates = copy.deepcopy(dict(app.state.config))
    app.state.model_active = app.state.config.detection.model.weights
    app.state.config_model = parse_json(BASE_PATH / "models" / app.state.config.detection.model.config)
    app.state.models = sorted([file.name for file in (BASE_PATH / "models").glob("*.blob")])
    app.state.scripts = sorted([file.name for file in BASE_PATH.glob("*.py")])
    app.state.logs = sorted([file.name for file in LOGS_PATH.glob("*.log")])
    app.state.configs = sorted([file.name for file in (BASE_PATH / "configs").glob("*.yaml")
                                if file.name != "config_selector.yaml"])

    # Initialize relevant app.state variables
    app.state.connection = get_current_connection()
    app.state.refresh_interval = max(round(1 / app.state.config.webapp.fps, 3), 0.033)  # max. 30 FPS
    app.state.labels = app.state.config_model.mappings.labels
    app.state.show_overlay = True
    app.state.last_overlay_empty = True
    app.state.exposure_region_active = False
    app.state.start_recording = False
    app.state.focus_initialized = False
    app.state.manual_focus_enabled = app.state.config.camera.focus.mode == "manual"
    app.state.focus_range_enabled = app.state.config.camera.focus.mode == "range"
    app.state.focus_distance_enabled = app.state.config.camera.focus.type == "distance"
    app.state.rec_durations = {
        "default": convert_duration(app.state.config.recording.duration.default),
        "battery": {level: convert_duration(getattr(app.state.config.recording.duration.battery, level))
                    for level in ["high", "medium", "low", "minimal"]}
    }
    app.state.aspect_ratio = app.state.config.webapp.resolution.width / app.state.config.webapp.resolution.height
    app.state.frame_count = 0
    app.state.fps = 0
    app.state.lens_pos = 0
    app.state.iso_sens = 0
    app.state.exp_time = 0
    app.state.prev_time = time.monotonic()

    # Create OAK camera pipeline and start device in USB2 mode
    pipeline, app.state.sensor_res = create_pipeline(BASE_PATH, app.state.config, app.state.config_model,
                                                     use_webapp_config=True, create_xin=True)
    app.state.device = dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH)

    # Create output queues to get the synchronized HQ frames and tracker + model output
    app.state.q_frame = app.state.device.getOutputQueue(name="frame", maxSize=2, blocking=False)
    app.state.q_track = app.state.device.getOutputQueue(name="track", maxSize=1, blocking=False)

    # Create input queue to send control commands to OAK camera
    app.state.q_ctrl = app.state.device.getInputQueue(name="control", maxSize=4, blocking=False)

    ui.notification("OAK camera pipeline started!", type="positive", timeout=2)


async def close_camera():
    """Stop streaming and disconnect from OAK device."""
    for queue in ("q_frame", "q_track", "q_ctrl"):
        if getattr(app.state, queue, None):
            setattr(app.state, queue, None)

    if getattr(app.state, "overlay_timer", None):
        app.state.overlay_timer.deactivate()
        app.state.overlay_timer = None

    if getattr(app.state, "device", None):
        app.state.device.close()
        app.state.device = None


def get_frame(q_frame):
    """Get MJPEG-encoded frame and associated metadata from the OAK camera output queue."""
    if not q_frame:
        return None
    try:
        frame_msg = q_frame.tryGet()  # depthai.ImgFrame (type: BITSTREAM)
        if frame_msg is None:
            return None
        frame_bytes = frame_msg.getData().tobytes()  # convert numpy array to bytes
        frame_bytes_length = str(len(frame_bytes)).encode()
        lens_pos = frame_msg.getLensPosition()
        iso_sens = frame_msg.getSensitivity()
        exp_time = frame_msg.getExposureTime().total_seconds() * 1000  # convert to milliseconds
        return (frame_bytes, frame_bytes_length, lens_pos, iso_sens, exp_time)
    except Exception:
        logger.exception("Error getting frame")
        return None


async def frame_generator():
    """Yield MJPEG-encoded frames asynchronously and update camera parameters."""
    try:
        next_tick = time.monotonic()
        while getattr(app.state, "q_frame", None):
            frame_data = await run.io_bound(get_frame, app.state.q_frame)

            if frame_data:
                frame_bytes, frame_bytes_length, lens_pos, iso_sens, exp_time = frame_data

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + frame_bytes_length + b"\r\n\r\n"
                       + frame_bytes + b"\r\n")

                # Update camera parameters twice per second
                app.state.frame_count += 1
                current_time = time.monotonic()
                elapsed_time = current_time - app.state.prev_time
                if elapsed_time > 0.5:
                    app.state.fps = round(app.state.frame_count / elapsed_time, 2)
                    app.state.lens_pos = lens_pos if lens_pos is not None else 0
                    app.state.iso_sens = iso_sens if iso_sens is not None else 0
                    app.state.exp_time = exp_time if exp_time is not None else 0
                    app.state.frame_count = 0
                    app.state.prev_time = current_time
            else:
                yield (b"--frame\r\n"
                       b"Content-Type: image/png\r\n"
                       b"Content-Length: " + PLACEHOLDER_PNG_BYTES_LENGTH + b"\r\n\r\n"
                       + PLACEHOLDER_PNG_BYTES + b"\r\n")

            next_tick += app.state.refresh_interval
            delay = next_tick - time.monotonic()
            # Reset next_tick if more than one interval late to avoid drift or frame bursts
            if delay < -app.state.refresh_interval:
                next_tick = time.monotonic()
            await asyncio.sleep(max(delay, 0))
    except asyncio.CancelledError:
        return


async def get_tracker_data():
    """Get model/tracker data from the OAK camera output queue, set exposure region if enabled."""
    tracker_data = []
    track_id_max = -1
    track_id_max_bbox = None

    if getattr(app.state, "q_track", None):
        tracker_msg = app.state.q_track.tryGet()
        if tracker_msg is None:
            return tracker_data
        tracklets = tracker_msg.tracklets
        for tracklet in tracklets:
            # Check if tracklet is active (not "LOST" or "REMOVED")
            tracklet_status = tracklet.status.name
            if tracklet_status in {"TRACKED", "NEW"}:
                track_id = tracklet.id
                bbox = (tracklet.srcImgDetection.xmin, tracklet.srcImgDetection.ymin,
                        tracklet.srcImgDetection.xmax, tracklet.srcImgDetection.ymax)

                if tracklet_status == "TRACKED" and track_id > track_id_max:
                    track_id_max = track_id
                    track_id_max_bbox = bbox

                tracklet_data = {
                    "label": app.state.labels[tracklet.srcImgDetection.label],
                    "confidence": round(tracklet.srcImgDetection.confidence, 2),
                    "track_ID": track_id,
                    "track_status": tracklet_status,
                    "x_min": round(bbox[0], 4),
                    "y_min": round(bbox[1], 4),
                    "x_max": round(bbox[2], 4),
                    "y_max": round(bbox[3], 4)
                }
                tracker_data.append(tracklet_data)

        if app.state.config_updates["detection"]["exposure_region"]["enabled"]:
            if track_id_max_bbox:
                # Use model bbox from most recent active tracking ID to set auto exposure region
                roi_x, roi_y, roi_w, roi_h = convert_bbox_roi(track_id_max_bbox, app.state.sensor_res)
                exp_ctrl = dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_w, roi_h)
                app.state.q_ctrl.send(exp_ctrl)
                app.state.exposure_region_active = True
            elif app.state.exposure_region_active:
                # Reset auto exposure region to full frame if there is no active tracking ID
                roi_x, roi_y, roi_w, roi_h = 1, 1, app.state.sensor_res[0] - 1, app.state.sensor_res[1] - 1
                exp_ctrl = dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_w, roi_h)
                app.state.q_ctrl.send(exp_ctrl)
                app.state.exposure_region_active = False

    return tracker_data


async def build_overlay(tracker_data):
    """Build SVG overlay with latest model/tracker data."""
    svg_parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1" width="100%" height="100%" '
        'style="position:absolute; top:0; left:0; pointer-events:none;">'
    ]

    for data in tracker_data:
        label = data["label"]
        confidence = data["confidence"]
        track_id = data["track_ID"]
        x_min = (data["x_min"] - 0.5) * app.state.aspect_ratio + 0.5
        y_min = data["y_min"]
        x_max = (data["x_max"] - 0.5) * app.state.aspect_ratio + 0.5
        y_max = data["y_max"]
        width = x_max - x_min
        height = y_max - y_min

        # Add rectangle for bounding box
        svg_parts.append(
            f'<rect x="{x_min}" y="{y_min}" width="{width}" height="{height}" '
            'fill="none" stroke="red" stroke-width="0.006" stroke-opacity="0.5" />'
        )

        # Add text for model/tracker data
        text_y = y_min + height + 0.04 if y_min + height < 0.95 else y_min - 0.05
        svg_parts.append(
            f'<text x="{x_min}" y="{text_y}" '
            'font-size="0.04" fill="white" stroke="black" stroke-width="0.005" '
            'paint-order="stroke" text-anchor="start" font-weight="bold">'
            f'{label} {confidence}'
            f'<tspan x="{x_min}" dy="0.04">ID: {track_id}</tspan></text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


async def update_overlay():
    """Get latest model/tracker data and update overlay."""
    if app.state.show_overlay or app.state.config_updates["detection"]["exposure_region"]["enabled"]:
        tracker_data = await get_tracker_data()
        if not tracker_data:
            if not getattr(app.state, "last_overlay_empty", False):
                app.state.frame_ii.set_content("")
                app.state.last_overlay_empty = True
            return
        if app.state.show_overlay:
            svg_overlay = await build_overlay(tracker_data)
            app.state.frame_ii.set_content(svg_overlay)
            app.state.last_overlay_empty = False
    else:
        if not getattr(app.state, "last_overlay_empty", False):
            app.state.frame_ii.set_content("")
            app.state.last_overlay_empty = True


async def set_manual_focus(e):
    """Set manual focus position of OAK camera."""
    if app.state.focus_initialized:
        mf_ctrl = dai.CameraControl().setManualFocus(e.value)
        app.state.q_ctrl.send(mf_ctrl)
    else:
        app.state.focus_initialized = True


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


async def on_config_change(e):
    """Switch to selected config file and apply new configuration parameters."""
    config_selected_name = e.value
    if config_selected_name == app.state.config_active:
        return

    has_unsaved_changes = check_config_changes(app.state.config, app.state.config_updates)
    if has_unsaved_changes:
        with ui.dialog() as dialog, ui.card():
            ui.label("You have unsaved configuration changes!").classes("text-h6 font-bold")
            ui.label("Do you want to save them before switching to a different config?")
            with ui.row().classes("w-full justify-center gap-4 mt-4"):
                ui.button("Cancel", on_click=lambda: dialog.submit("cancel"))
                ui.button("Save Config", on_click=lambda: dialog.submit("save"), color="green", icon="save")
                ui.button("Switch Without Saving", on_click=lambda: dialog.submit("proceed"), color="orange")

        action = await dialog
        if action == "cancel":
            app.state.config_select_ui.set_value(app.state.config_active)
            ui.notification("Configuration switch cancelled!", type="warning", timeout=2)
            return
        elif action == "save":
            app.state.config_select_ui.set_value(app.state.config_active)
            await save_config()
            return

    config_selected = parse_yaml(BASE_PATH / "configs" / config_selected_name)
    has_network_changes = check_config_changes(app.state.config.network,
                                               config_selected.network)

    await apply_config_changes(config_selected_name, has_network_changes, config_selected)


def create_control_elements():
    """Create UI elements and config binding for camera, web app and config control."""
    # Slider for manual focus control (only visible if focus mode is set to "manual")
    with ui.column().classes("w-full gap-0 mb-0").bind_visibility_from(app.state, "manual_focus_enabled"):
        ui.label("Manual Focus:").classes("font-bold")
        (ui.slider(min=0, max=255, step=1, on_change=set_manual_focus).props("label")
         .bind_value(app.state.config_updates["camera"]["focus"]["lens_position"], "manual"))

    # Slider for auto focus range control (only visible if focus mode is set to "range")
    with ui.column().classes("w-full gap-0 mb-0").bind_visibility_from(app.state, "focus_range_enabled"):
        ui.label("Focus Range:").classes("font-bold")
        (ui.range(min=0, max=255, step=1, on_change=preview_focus_range).props("label")
         .bind_value(app.state.config_updates["camera"]["focus"]["lens_position"], "range"))

    with ui.row(align_items="center").classes("w-full gap-2"):
        # Switches to toggle dark mode and model/tracker overlay
        (ui.switch("Dark", value=True).props("color=green").classes("font-bold")
         .bind_value_to(ui.dark_mode()))
        ui.separator().props("vertical")
        (ui.switch("Overlay").props("color=green").classes("font-bold")
         .bind_value(app.state, "show_overlay"))

        # WiFi/Hotspot status icons
        ui.separator().props("vertical")
        if app.state.connection["mode"] == "wifi":
            ui.icon("wifi", color="green")
            ui.icon("wifi_tethering_off", color="gray")
        elif app.state.connection["mode"] == "hotspot":
            ui.icon("wifi_off", color="gray")
            ui.icon("wifi_tethering", color="green")
        ui.label(f"{app.state.connection['ssid']}").classes("text-xs")

    # Config file selector
    with ui.row(align_items="center").classes("w-full gap-2 mt-0"):
        (ui.label("Active Config:").classes("font-bold whitespace-nowrap")
         .tooltip("Activate config file that will be used by the web app and recording script"))
        app.state.config_select_ui = ui.select(app.state.configs, value=app.state.config_active,
                                               on_change=on_config_change).classes("flex-1 truncate")


async def get_location():
    """Get current location using the Geolocation API and save to config."""
    try:
        response = await ui.run_javascript('''
            return await new Promise((resolve, reject) => {
                if ("geolocation" in navigator) {
                    const options = {
                        enableHighAccuracy: true,
                        timeout: 20000,
                        maximumAge: 0
                    };
                    navigator.geolocation.getCurrentPosition(
                        position => {
                            const result = {
                                latitude: position.coords.latitude,
                                longitude: position.coords.longitude
                            };
                            if (position.coords.accuracy != null) {
                                result.accuracy = position.coords.accuracy;
                            }
                            resolve(result);
                        },
                        error => reject(error),
                        options
                    );
                } else {
                    reject("Geolocation is not supported by this browser.");
                }
            });
        ''', timeout=25)

        if response is None:
            ui.notification("Location request failed. Please try again", type="warning", timeout=3)
            return None

        app.state.config_updates["deployment"]["location"]["latitude"] = response["latitude"]
        app.state.config_updates["deployment"]["location"]["longitude"] = response["longitude"]
        if "accuracy" in response:
            app.state.config_updates["deployment"]["location"]["accuracy"] = round(response["accuracy"])

    except TimeoutError:
        ui.notification("Location request timed out. Please try again", type="warning", timeout=3)
        return None


def create_deployment_section():
    """Create UI elements and config binding for deployment metadata."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        (ui.label("Start Time").classes("font-bold")
         .tooltip("Start date + time of the camera deployment (ISO 8601 format)"))
        with ui.row(align_items="center").classes("w-full gap-2"):
            time_label = (ui.label().classes("flex-1 min-h-8 py-2 px-3 rounded border border-gray-700")
                          .bind_text(app.state.config_updates["deployment"], "start"))
            ui.button("Get RPi Time", icon="event",
                      on_click=lambda: time_label.set_text(str(datetime.now().isoformat())))

        grid_separator()
        (ui.label("Location").classes("font-bold")
         .tooltip("Location of the camera deployment"))
        with ui.column().classes("w-full gap-2"):
            with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):
                ui.label("Latitude:").classes("font-bold")
                (ui.number(label="Latitude (decimal degrees)",
                           min=-90, max=90, precision=6, step=0.000001)
                 .bind_value(app.state.config_updates["deployment"]["location"], "latitude",
                             forward=lambda v: float(v) if v not in (None, "") else None))
                ui.label("Longitude:").classes("font-bold")
                (ui.number(label="Longitude (decimal degrees)",
                           min=-180, max=180, precision=6, step=0.000001)
                 .bind_value(app.state.config_updates["deployment"]["location"], "longitude",
                             forward=lambda v: float(v) if v not in (None, "") else None))
                ui.label("Accuracy:").classes("font-bold")
                (ui.number(label="Accuracy", min=0, max=1000, precision=1, step=1, suffix="m")
                 .bind_value(app.state.config_updates["deployment"]["location"], "accuracy",
                             forward=lambda v: int(v) if v not in (None, "") else None))
            loc_button = ui.button("Get Location", icon="my_location", on_click=get_location)
            if not app.state.config.webapp.https.enabled:
                loc_button.disable()
                loc_button.tooltip("HTTPS must be enabled for Geolocation API to work")

        grid_separator()
        (ui.label("Setting").classes("font-bold")
         .tooltip("Background setting of the camera (e.g. platform type/flower species)"))
        (ui.input(placeholder="Enter background setting").props("clearable")
         .bind_value(app.state.config_updates["deployment"], "setting",
                     forward=lambda v: str(v) if v is not None else None))

        grid_separator()
        (ui.label("Distance").classes("font-bold")
         .tooltip("Distance from camera to background (e.g. platform/flower)"))
        (ui.number(label="Distance", min=8, max=100, precision=0, step=1, suffix="cm",
                   validation={"Optional value between 8-100":
                               lambda v: v in (None, "") or validate_number(v, 8, 100)})
         .bind_value(app.state.config_updates["deployment"], "distance",
                     forward=lambda v: int(v) if v not in (None, "") else None))

        grid_separator()
        (ui.label("Notes").classes("font-bold")
         .tooltip("Additional notes about the deployment"))
        (ui.textarea(placeholder="Enter deployment notes").props("clearable")
         .bind_value(app.state.config_updates["deployment"], "notes",
                     forward=lambda v: str(v) if v is not None else None))


async def on_focus_mode_change(e):
    """Update relevant focus parameters in config, set continuous focus if selected."""
    app.state.manual_focus_enabled = e.value == "manual"
    app.state.focus_range_enabled = e.value == "range"
    if e.value == "continuous":
        af_ctrl = dai.CameraControl().setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        app.state.q_ctrl.send(af_ctrl)
    else:
        app.state.focus_initialized = False
        app.state.focus_distance_enabled = False
        app.state.config_updates["camera"]["focus"]["type"] = "lens_position"


async def on_focus_type_change(e):
    """Update focus distance visibility when focus type changes."""
    app.state.focus_distance_enabled = e.value == "distance"
    if e.value == "distance":
        ui.notification("Focus control slider will still use lens position for finer adjustment!",
                        type="warning", timeout=3)


def create_camera_settings():
    """Create UI elements and config binding for camera settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        ui.label("Focus Mode").classes("font-bold")
        (ui.select(["continuous", "manual", "range"], label="Mode", on_change=on_focus_mode_change)
         .bind_value(app.state.config_updates["camera"]["focus"], "mode"))

        grid_separator()
        ui.label("Focus Type").classes("font-bold")
        with ui.column().classes("w-full gap-1"):
            (ui.select(["distance", "lens_position"], label="Type", on_change=on_focus_type_change)
             .classes("w-full")
             .bind_value(app.state.config_updates["camera"]["focus"], "type"))

            with (ui.column().classes("w-full gap-1")
                  .bind_visibility_from(app.state, "focus_distance_enabled")):
                with ui.row(align_items="center").classes("w-full gap-2"):
                    (ui.number(label="Manual Focus",
                               placeholder=app.state.config.camera.focus.distance.manual,
                               min=8, max=80, precision=0, step=1, suffix="cm").classes("flex-1")
                     .bind_value(app.state.config_updates["camera"]["focus"]["distance"], "manual",
                                 forward=lambda v: int(v) if v is not None else None))
                with ui.row(align_items="center").classes("w-full gap-2"):
                    (ui.number(label="Range Min",
                               placeholder=app.state.config.camera.focus.distance.range.min,
                               min=8, max=75, precision=0, step=1, suffix="cm").classes("flex-1")
                     .bind_value(app.state.config_updates["camera"]["focus"]["distance"]["range"], "min",
                                 forward=lambda v: int(v) if v is not None else None))
                    (ui.number(label="Range Max",
                               placeholder=app.state.config.camera.focus.distance.range.max,
                               min=9, max=80, precision=0, step=1, suffix="cm").classes("flex-1")
                     .bind_value(app.state.config_updates["camera"]["focus"]["distance"]["range"], "max",
                                 forward=lambda v: int(v) if v is not None else None))
                (ui.label("Focus control slider will still use lens position for finer adjustment!")
                 .classes("text-xs text-gray-500"))

        grid_separator()
        ui.label("Frame Rate").classes("font-bold").tooltip("Higher FPS increases power consumption")
        (ui.number(label="FPS", placeholder=app.state.config.camera.fps,
                   min=1, max=30, precision=0, step=1,
                   validation={"Required value between 1-30": lambda v: validate_number(v, 1, 30)})
         .bind_value(app.state.config_updates["camera"], "fps",
                     forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        ui.label("Resolution").classes("font-bold").tooltip("Resolution of captured images (HQ frames)")
        with ui.row(align_items="center").classes("w-full gap-2"):
            (ui.number(label="Width", placeholder=app.state.config.camera.resolution.width,
                       min=320, max=3840, precision=0, step=32,
                       validation={"Required value between 320-3840 (multiple of 32)":
                                   lambda v: validate_number(v, 320, 3840, 32)}).classes("flex-1")
             .bind_value(app.state.config_updates["camera"]["resolution"], "width",
                         forward=lambda v: int(v) if v is not None else None))
            (ui.number(label="Height", placeholder=app.state.config.camera.resolution.height,
                       min=320, max=2160, precision=0, step=2,
                       validation={"Required value between 320-2160 (multiple of 2)":
                                   lambda v: validate_number(v, 320, 2160, 2)}).classes("flex-1")
             .bind_value(app.state.config_updates["camera"]["resolution"], "height",
                         forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        ui.label("JPEG Quality").classes("font-bold").tooltip("JPEG quality of captured images")
        (ui.number(label="JPEG", placeholder=app.state.config.camera.jpeg_quality,
                   min=10, max=100, precision=0, step=1,
                   validation={"Required value between 10-100": lambda v: validate_number(v, 10, 100)})
         .bind_value(app.state.config_updates["camera"], "jpeg_quality",
                     forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("ISP Settings").classes("font-bold")
         .tooltip("Setting Sharpness and Luma Denoise to 0 can reduce artifacts"))
        with ui.row(align_items="center").classes("w-full gap-2"):
            (ui.number(label="Sharpness", placeholder=app.state.config.camera.isp.sharpness,
                       min=0, max=4, precision=0, step=1,
                       validation={"Required value between 0-4": lambda v: validate_number(v, 0, 4)})
             .classes("flex-1")
             .bind_value(app.state.config_updates["camera"]["isp"], "sharpness",
                         forward=lambda v: int(v) if v is not None else None))
            (ui.number(label="Luma Denoise", placeholder=app.state.config.camera.isp.luma_denoise,
                       min=0, max=4, precision=1, step=1,
                       validation={"Required value between 0-4": lambda v: validate_number(v, 0, 4)})
             .classes("flex-1")
             .bind_value(app.state.config_updates["camera"]["isp"], "luma_denoise",
                         forward=lambda v: int(v) if v is not None else None))
            (ui.number(label="Chroma Denoise", placeholder=app.state.config.camera.isp.chroma_denoise,
                       min=0, max=4, precision=1, step=1,
                       validation={"Required value between 0-4": lambda v: validate_number(v, 0, 4)})
             .classes("flex-1")
             .bind_value(app.state.config_updates["camera"]["isp"], "chroma_denoise",
                         forward=lambda v: int(v) if v is not None else None))


async def on_exposure_region_change(e):
    """Reset auto exposure region to full frame if setting is disabled."""
    if not e.value and app.state.exposure_region_active:
        roi_x, roi_y, roi_w, roi_h = 1, 1, app.state.sensor_res[0] - 1, app.state.sensor_res[1] - 1
        exp_ctrl = dai.CameraControl().setAutoExposureRegion(roi_x, roi_y, roi_w, roi_h)
        app.state.q_ctrl.send(exp_ctrl)
        app.state.exposure_region_active = False


def create_detection_settings():
    """Create UI elements and config binding for detection settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        (ui.label("Detection-based Exposure").classes("font-bold")
         .tooltip("Use bounding box from most recent tracking ID to set auto exposure region"))
        (ui.switch("Enable", on_change=on_exposure_region_change).props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["detection"]["exposure_region"], "enabled"))

        grid_separator()
        ui.label("Detection Model").classes("font-bold")
        (ui.select(app.state.models, label="Model", value=app.state.model_active).classes("truncate")
         .bind_value(app.state.config_updates["detection"]["model"], "weights")
         .bind_value_to(app.state.config_updates["detection"]["model"], "config",
                        forward=lambda v: f"{Path(v).stem}.json" if v else None))

        grid_separator()
        (ui.label("Input Resolution").classes("font-bold")
         .tooltip("Resolution of downscaled + stretched/cropped LQ frames for model input"))
        with ui.row(align_items="center").classes("w-full gap-2"):
            (ui.number(label="Width", placeholder=app.state.config.detection.resolution.width,
                       min=128, max=640, precision=0, step=1,
                       validation={"Required value between 128-640":
                                   lambda v: validate_number(v, 128, 640)}).classes("flex-1")
             .bind_value(app.state.config_updates["detection"]["resolution"], "width",
                         forward=lambda v: int(v) if v is not None else None))
            (ui.number(label="Height", placeholder=app.state.config.detection.resolution.height,
                       min=128, max=640, precision=0, step=1,
                       validation={"Required value between 128-640":
                                   lambda v: validate_number(v, 128, 640)}).classes("flex-1")
             .bind_value(app.state.config_updates["detection"]["resolution"], "height",
                         forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        ui.label("Confidence Threshold").classes("font-bold").tooltip("Overrides model config file")
        (ui.number(label="Confidence", placeholder=app.state.config.detection.conf_threshold,
                   min=0, max=1, precision=2, step=0.01,
                   validation={"Required value between 0-1": lambda v: validate_number(v, 0, 1)})
         .bind_value(app.state.config_updates["detection"], "conf_threshold"))

        grid_separator()
        ui.label("IoU Threshold").classes("font-bold").tooltip("Overrides model config file")
        (ui.number(label="IoU", placeholder=app.state.config.detection.iou_threshold,
                   min=0, max=1, precision=2, step=0.01,
                   validation={"Required value between 0-1": lambda v: validate_number(v, 0, 1)})
         .bind_value(app.state.config_updates["detection"], "iou_threshold"))


def create_recording_settings():
    """Create UI elements and config binding for recording settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        ui.label("Duration").classes("font-bold").tooltip("Duration per recording session")
        with ui.column().classes("w-full"):
            with ui.tabs().classes("w-full") as tabs:
                ui.tab("Battery", icon="battery_charging_full")
                ui.tab("No Battery", icon="timer")
            with ui.tab_panels(tabs, value="Battery").classes("w-full"):
                with ui.tab_panel("Battery"):
                    create_duration_inputs("high", "High (> 70% or USB connected)",
                        "Duration if battery charge level is > 70% or USB power is connected")
                    create_duration_inputs("medium", "Medium (50-70%)",
                        "Duration if battery charge level is between 50-70%")
                    create_duration_inputs("low", "Low (30-50%)",
                        "Duration if battery charge level is between 30-50%")
                    create_duration_inputs("minimal", "Minimal (< 30%)",
                        "Duration if battery charge level is < 30%",)
                with ui.tab_panel("No Battery"):
                    create_duration_inputs("default", "Default",
                        "Duration if powermanager is disabled")

        grid_separator()
        ui.label("Capture Interval").classes("font-bold")
        with ui.column().classes("w-full"):
            with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):
                (ui.label("Detection").classes("font-bold")
                 .tooltip("Interval for saving HQ frame + metadata while object is detected"))
                (ui.number(label="Capture Interval",
                           placeholder=app.state.config.recording.capture_interval.detection,
                           min=0, max=3600, precision=1, step=0.1, suffix="seconds",
                           validation={"Required value between 0-3600":
                                       lambda v: validate_number(v, 0, 3600)})
                 .bind_value(app.state.config_updates["recording"]["capture_interval"], "detection"))
                (ui.label("Timelapse").classes("font-bold")
                 .tooltip("Interval for saving HQ frame (independent of detected objects)"))
                (ui.number(label="Capture Interval",
                           placeholder=app.state.config.recording.capture_interval.timelapse,
                           min=0, max=3600, precision=1, step=0.1, suffix="seconds",
                           validation={"Required value between 0-3600":
                                       lambda v: validate_number(v, 0, 3600)})
                 .bind_value(app.state.config_updates["recording"]["capture_interval"], "timelapse"))

        grid_separator()
        (ui.label("Shutdown After Recording").classes("font-bold")
         .tooltip("Shut down Raspberry Pi after recording session is finished or interrupted"))
        (ui.switch("Enable").props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["recording"]["shutdown"], "enabled"))


def create_processing_settings():
    """Create UI elements and config binding for post-processing settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        (ui.label("Crop Detections").classes("font-bold")
         .tooltip("Crop detections from HQ frames and save as individual .jpg images"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["post_processing"]["crop"], "enabled"))
            (ui.select(["square", "original"], label="Crop Method").classes("w-full")
             .bind_visibility_from(app.state.config_updates["post_processing"]["crop"], "enabled")
             .bind_value(app.state.config_updates["post_processing"]["crop"], "method"))

        grid_separator()
        (ui.label("Draw Overlays").classes("font-bold")
         .tooltip("Draw overlays on HQ frame copies (bounding box, label, confidence, track ID)"))
        (ui.switch("Enable").props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["post_processing"]["overlay"], "enabled"))

        grid_separator()
        (ui.label("Delete Originals").classes("font-bold")
         .tooltip("Delete original HQ frames with detections after processing"))
        (ui.switch("Enable").props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["post_processing"]["delete"], "enabled"))

        grid_separator()
        (ui.label("Archive Data").classes("font-bold")
         .tooltip("Archive (zip) all captured data + logs/configs and manage disk space"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["archive"], "enabled"))
            (ui.number(label="Low Free Space", placeholder=app.state.config.archive.disk_low,
                       min=100, max=50000, precision=0, step=100, suffix="MB",
                       validation={"Required value between 100-50000 MB":
                                   lambda v: validate_number(v, 100, 50000)}).classes("w-full")
             .tooltip("Minimum required free disk space for unarchived data retention")
             .bind_visibility_from(app.state.config_updates["archive"], "enabled")
             .bind_value(app.state.config_updates["archive"], "disk_low",
                         forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("Upload to Cloud").classes("font-bold")
         .tooltip("Upload archived data to cloud storage provider (always runs archive)"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["upload"], "enabled"))
            (ui.select(["all", "full", "crop", "metadata"], label="Content").classes("w-full")
             .tooltip("Select content for upload, always including metadata")
             .bind_visibility_from(app.state.config_updates["upload"], "enabled")
             .bind_value(app.state.config_updates["upload"], "content"))


def create_startup_settings():
    """Create UI elements and config binding for startup settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        (ui.label("Hotspot Setup").classes("font-bold")
         .tooltip("Create RPi Wi-Fi hotspot if it doesn't exist (uses hostname for SSID and password)"))
        (ui.switch("Enable").props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["startup"]["hotspot_setup"], "enabled"))

        grid_separator()
        (ui.label("Network Setup").classes("font-bold")
         .tooltip("Create/update all configured Wi-Fi profiles in NetworkManager (including hotspot)"))
        (ui.switch("Enable").props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["startup"]["network_setup"], "enabled"))

        grid_separator()
        (ui.label("Auto Run").classes("font-bold")
         .tooltip("Automatically run configured Python script(s) after boot"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["startup"]["auto_run"], "enabled"))

            with (ui.column().classes("w-full")
                  .bind_visibility_from(app.state.config_updates["startup"]["auto_run"], "enabled")):
                (ui.select(app.state.scripts, label="Primary Script").classes("w-full truncate")
                 .tooltip("Primary Python script in 'insect-detect' directory that is run first")
                 .bind_value(app.state.config_updates["startup"]["auto_run"], "primary"))
                (ui.select(["None"] + app.state.scripts, label="Fallback Script").classes("w-full truncate")
                 .tooltip("Fallback Python script in 'insect-detect' directory (can be None)")
                 .bind_value(app.state.config_updates["startup"]["auto_run"], "fallback",
                             forward=lambda v: None if v == "None" else v,
                             backward=lambda v: "None" if v is None or v == "" else v))
                (ui.number(label="Delay", placeholder=app.state.config.startup.auto_run.delay,
                           min=1, max=1800, precision=0, step=1, suffix="seconds",
                           validation={"Required value between 1-1800":
                                       lambda v: validate_number(v, 1, 1800)}).classes("w-full")
                 .tooltip("Wait time before stopping primary script and running fallback script")
                 .bind_value(app.state.config_updates["startup"]["auto_run"], "delay",
                             forward=lambda v: int(v) if v is not None else None))


def create_webapp_settings():
    """Create UI elements and config binding for web app settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        (ui.label("Frame Rate").classes("font-bold")
         .tooltip("Max. possible streamed FPS depends on resolution"))
        (ui.number(label="FPS", placeholder=app.state.config.webapp.fps,
                   min=1, max=30, precision=0, step=1,
                   validation={"Required value between 1-30": lambda v: validate_number(v, 1, 30)})
         .bind_value(app.state.config_updates["webapp"], "fps",
                     forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        ui.label("Resolution").classes("font-bold").tooltip("Resolution of streamed HQ frames")
        with ui.row(align_items="center").classes("w-full gap-2"):
            (ui.number(label="Width", placeholder=app.state.config.webapp.resolution.width,
                       min=320, max=1920, precision=0, step=32,
                       validation={"Required value between 320-1920 (multiple of 32)":
                                   lambda v: validate_number(v, 320, 1920, 32)}).classes("flex-1")
             .bind_value(app.state.config_updates["webapp"]["resolution"], "width",
                         forward=lambda v: int(v) if v is not None else None))
            (ui.number(label="Height", placeholder=app.state.config.webapp.resolution.height,
                       min=320, max=1080, precision=0, step=2,
                       validation={"Required value between 320-1080 (multiple of 2)":
                                   lambda v: validate_number(v, 320, 1080, 2)}).classes("flex-1")
             .bind_value(app.state.config_updates["webapp"]["resolution"], "height",
                         forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        ui.label("JPEG Quality").classes("font-bold").tooltip("JPEG quality of streamed HQ frames")
        (ui.number(label="JPEG", placeholder=app.state.config.webapp.jpeg_quality,
                   min=10, max=100, precision=0, step=1,
                   validation={"Required value between 10-100":
                               lambda v: validate_number(v, 10, 100)})
         .bind_value(app.state.config_updates["webapp"], "jpeg_quality",
                     forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("Use HTTPS").classes("font-bold")
         .tooltip("Use HTTPS protocol (required for browser Geolocation API to get GPS location)"))
        (ui.switch("Enable", on_change=lambda e: ui.notification(
            "Protocol changes require a full web app restart to take effect.", type="warning", timeout=3)
            if e.value != app.state.config.webapp.https.enabled else None)
         .props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["webapp"]["https"], "enabled"))


def create_system_settings():
    """Create UI elements and config binding for system settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        (ui.label("Power Management").classes("font-bold")
         .tooltip("Disable if no power management board is connected"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["powermanager"], "enabled"))

            with (ui.column().classes("w-full")
                  .bind_visibility_from(app.state.config_updates["powermanager"], "enabled")):
                (ui.select(["wittypi", "pijuice"], label="Board Model").classes("w-full")
                 .bind_value(app.state.config_updates["powermanager"], "model"))
                with ui.row(align_items="center").classes("w-full gap-2"):
                    (ui.number(label="Min. Charge", placeholder=app.state.config.powermanager.charge_min,
                               min=10, max=90, precision=0, step=5, suffix="%",
                               validation={"Required value between 10-90":
                                           lambda v: validate_number(v, 10, 90)}).classes("flex-1")
                     .tooltip("Minimum required charge level to start/continue a recording")
                     .bind_value(app.state.config_updates["powermanager"], "charge_min",
                                 forward=lambda v: int(v) if v is not None else None))
                    (ui.number(label="Check Interval", placeholder=app.state.config.powermanager.charge_check,
                               min=5, max=300, precision=0, step=5, suffix="seconds",
                               validation={"Required value between 5-300":
                                           lambda v: validate_number(v, 5, 300)}).classes("flex-1")
                     .bind_value(app.state.config_updates["powermanager"], "charge_check",
                                 forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("OAK Temperature").classes("font-bold")
         .tooltip("Maximum allowed OAK chip temperature to continue a recording"))
        with ui.row(align_items="center").classes("w-full gap-2"):
            (ui.number(label="Max. Temperature", placeholder=app.state.config.oak.temp_max,
                       min=70, max=100, precision=0, step=1, suffix="°C",
                       validation={"Required value between 70-100":
                                   lambda v: validate_number(v, 70, 100)}).classes("flex-1")
             .bind_value(app.state.config_updates["oak"], "temp_max",
                         forward=lambda v: int(v) if v is not None else None))
            (ui.number(label="Check Interval", placeholder=app.state.config.oak.temp_check,
                       min=5, max=300, precision=0, step=5, suffix="seconds",
                       validation={"Required value between 5-300":
                                   lambda v: validate_number(v, 5, 300)}).classes("flex-1")
             .bind_value(app.state.config_updates["oak"], "temp_check",
                         forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        ui.label("Storage Management").classes("font-bold")
        with ui.row(align_items="center").classes("w-full gap-2"):
            (ui.number(label="Min. Free Space", placeholder=app.state.config.storage.disk_min,
                       min=100, max=10000, precision=0, step=100, suffix="MB",
                       validation={"Required value between 100-10000 MB":
                                   lambda v: validate_number(v, 100, 10000)}).classes("flex-1")
             .tooltip("Minimum required free disk space to start/continue a recording")
             .bind_value(app.state.config_updates["storage"], "disk_min",
                         forward=lambda v: int(v) if v is not None else None))
            (ui.number(label="Check Interval", placeholder=app.state.config.storage.disk_check,
                       min=5, max=300, precision=0, step=5, suffix="seconds",
                       validation={"Required value between 5-300":
                                   lambda v: validate_number(v, 5, 300)}).classes("flex-1")
             .bind_value(app.state.config_updates["storage"], "disk_check",
                         forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("Status LED").classes("font-bold")
         .tooltip("Use LED (e.g. in button) to indicate system status"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["led"], "enabled"))
            (ui.select(LED_GPIO_PINS, label="GPIO Pin (BCM)").classes("w-full")
             .bind_visibility_from(app.state.config_updates["led"], "enabled")
             .bind_value(app.state.config_updates["led"], "gpio_pin",
                         forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("System Logging").classes("font-bold")
         .tooltip("Log system information (temperature, memory, CPU utilization, battery info)"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["logging"], "enabled"))
            (ui.number(label="Log Interval", placeholder=app.state.config.logging.interval,
                       min=1, max=600, precision=0, step=1, suffix="seconds",
                       validation={"Required value between 1-600":
                                   lambda v: validate_number(v, 1, 600)}).classes("w-full")
             .bind_visibility_from(app.state.config_updates["logging"], "enabled")
             .bind_value(app.state.config_updates["logging"], "interval",
                         forward=lambda v: int(v) if v is not None else None))


def remove_wifi_network(network_row):
    """Remove a specific network row from UI and config"""
    if network_row in app.state.wifi_networks_ui:
        if len(app.state.wifi_networks_ui) > 1:
            idx = app.state.wifi_networks_ui.index(network_row)

            if idx < len(app.state.config_updates["network"]["wifi"]):
                app.state.config_updates["network"]["wifi"].pop(idx)

            network_row.delete()
            app.state.wifi_networks_ui.pop(idx)
        else:
            ui.notification("At least one Wi-Fi network must be configured!", type="warning", timeout=2)


def add_wifi_network(networks_column, ssid="", password=""):
    """Add a new Wi-Fi network input field."""
    with networks_column:
        new_network = {"ssid": ssid, "password": password}
        app.state.config_updates["network"]["wifi"].append(new_network)
        idx = len(app.state.config_updates["network"]["wifi"]) - 1

        with ui.row(align_items="baseline").classes("w-full gap-2") as network_row:
            (ui.input(label="SSID").props("clearable").classes("flex-1")
             .bind_value(app.state.config_updates["network"]["wifi"][idx], "ssid",
                         forward=lambda v: str(v) if v is not None else None))
            (ui.input(label="Password", validation={
                "Minimum 8 characters": lambda v: v is None or v == "" or len(str(v)) >= 8})
             .props("clearable").classes("flex-1")
             .bind_value(app.state.config_updates["network"]["wifi"][idx], "password",
                         forward=lambda v: str(v) if v is not None else None))
            ui.button(color="red", icon="delete",
                      on_click=lambda: remove_wifi_network(network_row)).props("round")

    app.state.wifi_networks_ui.append(network_row)


def create_network_settings():
    """Create UI elements and config binding for network settings."""
    app.state.wifi_networks_ui = []
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        ui.label("Mode").classes("font-bold").tooltip("Network mode of the Raspberry Pi")
        (ui.select(["hotspot", "wifi"], label="Network Mode").classes("w-full")
         .bind_value(app.state.config_updates["network"], "mode"))

        grid_separator()
        ui.label("RPi Hotspot").classes("font-bold")
        with ui.column().classes("w-full"):
            with ui.row(align_items="baseline").classes("w-full gap-2"):
                (ui.input(label="SSID", placeholder=HOSTNAME).props("clearable").classes("flex-1")
                 .bind_value(app.state.config_updates["network"]["hotspot"], "ssid",
                             forward=lambda v: str(v) if v is not None else None))
                (ui.input(label="Password", validation={
                    "Minimum 8 characters": lambda v: v is None or v == "" or len(str(v)) >= 8})
                 .props("clearable").classes("flex-1")
                 .bind_value(app.state.config_updates["network"]["hotspot"], "password",
                             forward=lambda v: str(v) if v is not None else None))

        grid_separator()
        (ui.label("Wi-Fi Networks").classes("font-bold")
         .tooltip("List of Wi-Fi networks that the RPi should connect to (ordered by priority)"))
        wifi_column = ui.column().classes("w-full gap-2")

        with wifi_column:
            app.state.config_updates["network"]["wifi"].clear()
            networks_column = ui.column().classes("w-full gap-2")
            for network in app.state.config.network.wifi:
                add_wifi_network(networks_column, network["ssid"], network["password"])
            ui.button("Add Wi-Fi", color="green", icon="add",
                      on_click=lambda: add_wifi_network(networks_column))


def read_split_log(log_path, max_lines=1000):
    """Read log file and return the last requested lines."""
    with open(log_path, "r", encoding="utf-8") as log_file:
        return list(deque(log_file, maxlen=max_lines))


async def update_log_content(selected_log, log_display):
    """Update content of log element based on selected log file."""
    log_display.clear()

    if not selected_log:
        return

    try:
        log_lines = await run.io_bound(read_split_log, LOGS_PATH / selected_log, max_lines=1000)
    except Exception as e:
        log_display.push(f"Error reading log file: {str(e)}", classes="text-red")
        return

    if not log_lines:
        log_display.push("Log file is empty", classes="text-gray")
        return

    for idx, line in enumerate(log_lines):
        lower = line.lower()
        if "error" in lower or "exception" in lower:
            log_display.push(line, classes="text-red")
        elif "warning" in lower:
            log_display.push(line, classes="text-orange")
        else:
            log_display.push(line)
        if (idx + 1) % 50 == 0:
            await asyncio.sleep(0)  # yield control back to the event loop to avoid blocking of UI


def create_logs_section():
    """Create UI elements for selecting and viewing log files."""
    with ui.column().classes("w-full gap-2"):
        if not app.state.logs:
            ui.label(f"No .log files found in '{LOGS_PATH}'").classes("text-gray")
            return

        log_select_ui = (ui.select(app.state.logs, label="Log File", value=None,
                                   on_change=lambda e: update_log_content(e.value, log_display))
                         .classes("w-full truncate"))

        log_display = (ui.log(max_lines=1000).classes("w-full h-96 font-mono text-xs")
                       .bind_visibility_from(log_select_ui, "value",
                                             backward=lambda v: v is not None and v != ""))


async def apply_config_changes(config_name, has_network_changes, config_selected=None):
    """Apply network changes, update config selector and restart the camera with new config."""
    if has_network_changes:
        with ui.dialog() as dialog, ui.card():
            ui.label("Network Configuration Change").classes("text-h6 font-bold")
            ui.label("Applying network configuration changes will interrupt your connection.")
            ui.label("You will probably need to connect to a different network afterwards.")
            ui.label("Do you want to continue?")

            with ui.row().classes("w-full justify-center gap-4 mt-4"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False))
                ui.button("Apply Network Changes", on_click=lambda: dialog.submit(True),
                          color="orange", icon="warning")

        apply_network_changes = await dialog
        if not apply_network_changes:
            if config_selected is not None:
                app.state.config_select_ui.set_value(app.state.config_active)
                ui.notification("Configuration not switched!", type="warning", timeout=2)
            else:
                ui.notification("Configuration not applied!", type="warning", timeout=2)
            return

        ui.notification("Applying network changes in 3 seconds...",
                        position="top", type="info", spinner=True, timeout=3)
        await asyncio.sleep(3)

        try:
            if config_selected is not None:
                set_up_network(dict(config_selected), activate_network=True)
            else:
                set_up_network(app.state.config_updates, activate_network=True)
        except Exception as e:
            ui.notification(f"Network settings failed to apply: {str(e)}", type="negative", timeout=5)
            return

    ui.notification(f"Activating configuration '{config_name}'...",
                    position="top", type="info", spinner=True, timeout=2)
    await asyncio.sleep(0.5)
    update_config_selector(BASE_PATH, config_name)
    app.state.config_active = config_name
    await close_camera()
    await asyncio.sleep(0.5)
    ui.navigate.reload()


async def show_apply_dialog(config_name, has_network_changes):
    """Show dialog to apply changes to current config."""
    with ui.dialog() as dialog, ui.card():
        ui.label(f"Configuration '{config_name}' has been updated.")
        ui.label("Do you want to apply the changes now?")

        with ui.row().classes("w-full justify-center gap-4 mt-4"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False))
            ui.button("Apply Changes", on_click=lambda: dialog.submit(True), color="green")

    apply_changes = await dialog
    if apply_changes:
        await apply_config_changes(config_name, has_network_changes)
    else:
        if has_network_changes:
            ui.notification(
                "Configuration saved but not applied yet! Please apply it to activate your network changes.",
                type="warning", timeout=3)
        else:
            ui.notification(
                "Configuration saved but not applied yet! Will be used for next web app or recording start.",
                type="info", timeout=3)


async def show_activate_dialog(config_name, has_network_changes):
    """Show dialog to activate another config."""
    with ui.dialog() as dialog, ui.card():
        ui.label(f"Configuration saved to '{config_name}'")
        ui.label("Do you want to activate this configuration now?")

        with ui.row().classes("w-full justify-center gap-4 mt-4"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False))
            ui.button("Activate Config", on_click=lambda: dialog.submit(True), color="green")

    activate_config = await dialog
    if activate_config:
        await apply_config_changes(config_name, has_network_changes)
    else:
        # Refresh all UI elements to reflect the still active config (reset config_updates)
        ui.notification("Configuration not activated!", type="warning", timeout=2)
        await asyncio.sleep(0.5)
        create_ui_layout.refresh()


async def save_to_file(config_path):
    """Save configuration to specified file path."""
    has_network_changes = check_config_changes(app.state.config.network,
                                               app.state.config_updates["network"])

    config_template_path = BASE_PATH / "configs" / app.state.config_active
    update_config_file(config_path, config_template_path, app.state.config_updates,
                       dict(app.state.config), OPTIONAL_CONFIG_FIELDS)

    ui.notification(f"Configuration saved to '{config_path.name}'!", type="positive", timeout=2)

    app.state.configs = sorted([file.name for file in (BASE_PATH / "configs").glob("*.yaml")
                                if file.name != "config_selector.yaml"])

    if config_path.name == app.state.config_active:
        # Update currently loaded config if saving to same config file
        app.state.config = parse_yaml(config_path)
        await show_apply_dialog(config_path.name, has_network_changes)
    else:
        # Reset config updates if saving to a different config file
        app.state.config_updates = copy.deepcopy(dict(app.state.config))
        await show_activate_dialog(config_path.name, has_network_changes)


async def config_name_input():
    """Show dialog to enter a name for the new configuration file."""
    with ui.dialog() as dialog, ui.card():
        ui.label("Name for new config file:")
        i = (ui.input(placeholder="config_custom",
                      validation={"Please enter a valid filename":
                                  lambda v: v is not None and all(c.isalnum() or c in "_-" for c in v)})
             .props("clearable autofocus suffix='.yaml'"))

        with ui.row().classes("w-full justify-center gap-4 mt-4"):
            ui.button("Cancel", on_click=lambda: dialog.submit("cancel"))
            ui.button("Save", on_click=lambda: dialog.submit(i.value), color="green")

    return await dialog


async def create_new_config():
    """Create a new configuration file."""
    filename = await config_name_input()
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

        action = await dialog
        if action == "cancel":
            ui.notification("New config creation cancelled!", type="warning", timeout=2)
            return
        if action == "new_name":
            await create_new_config()
            return

    await save_to_file(config_new_path)


async def save_config():
    """Save configuration while preserving comments and structure."""
    if app.state.config_active == "config_default.yaml":
        ui.notification("Cannot save changes to default configuration!", type="warning", timeout=2)
        await create_new_config()
        return

    config_current_path = BASE_PATH / "configs" / app.state.config_active

    with ui.dialog() as dialog, ui.card():
        ui.label(f"Save changes to '{app.state.config_active}'?")
        with ui.row().classes("w-full justify-center gap-4 mt-4"):
            ui.button("Cancel", on_click=lambda: dialog.submit("cancel"))
            ui.button("Create New", on_click=lambda: dialog.submit("new"), color="green")
            ui.button("Overwrite", on_click=lambda: dialog.submit("overwrite"), color="orange")

    action = await dialog
    if action == "cancel":
        ui.notification("Changes not saved!", type="warning", timeout=2)
    elif action == "new":
        await create_new_config()
    elif action == "overwrite":
        await save_to_file(config_current_path)


async def start_recording():
    """Launch the recording script after shutting down the web app."""
    has_unsaved_changes = check_config_changes(app.state.config, app.state.config_updates)
    if has_unsaved_changes:
        with ui.dialog() as dialog, ui.card():
            ui.label("You have unsaved configuration changes!").classes("text-h6 font-bold")
            ui.label("Do you want to save them before starting the recording?")
            with ui.row().classes("w-full justify-center gap-4 mt-4"):
                ui.button("Cancel", on_click=lambda: dialog.submit("cancel"))
                ui.button("Save Config", on_click=lambda: dialog.submit("save"), color="green", icon="save")
                ui.button("Start Without Saving", on_click=lambda: dialog.submit("proceed"), color="orange")

        action = await dialog
        if action == "cancel":
            ui.notification("Recording start cancelled!", type="warning", timeout=2)
            return
        elif action == "save":
            await save_config()
            return

    with ui.dialog() as dialog, ui.card():
        ui.label("Are you sure you want to stop the web app and start the recording script?")
        with ui.row().classes("w-full justify-center gap-4 mt-4"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False))
            ui.button("Start Recording", on_click=lambda: dialog.submit(True), color="teal", icon="play_circle")

    start_rec = await dialog
    if start_rec:
        app.state.start_recording = True
        ui.notification("Stopping web app and start recording...",
                        position="top", type="ongoing", spinner=True, timeout=3)
        await asyncio.sleep(0.5)
        await close_camera()
        app.shutdown()


async def confirm_shutdown():
    """Confirm or cancel shutdown of the web app."""
    has_unsaved_changes = check_config_changes(app.state.config, app.state.config_updates)
    if has_unsaved_changes:
        with ui.dialog() as dialog, ui.card():
            ui.label("You have unsaved configuration changes!").classes("text-h6 font-bold")
            ui.label("Do you want to save them before stopping the web app?")
            with ui.row().classes("w-full justify-center gap-4 mt-4"):
                ui.button("Cancel", on_click=lambda: dialog.submit("cancel"))
                ui.button("Save Config", on_click=lambda: dialog.submit("save"), color="green", icon="save")
                ui.button("Stop Without Saving", on_click=lambda: dialog.submit("proceed"), color="orange")

        action = await dialog
        if action == "cancel":
            ui.notification("Web App Shutdown cancelled!", type="warning", timeout=2)
            return
        elif action == "save":
            await save_config()
            return

    with ui.dialog() as dialog, ui.card():
        ui.label("Are you sure you want to stop the web app?")
        with ui.row().classes("w-full justify-center gap-4 mt-4"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False))
            ui.button("Stop App", on_click=lambda: dialog.submit(True), color="red", icon="power_settings_new")

    shutdown = await dialog
    if shutdown:
        ui.notification("Stopping web app...", position="top", type="ongoing", spinner=True, timeout=3)
        await asyncio.sleep(0.5)
        await close_camera()
        app.shutdown()


async def disconnect():
    """Disconnect all clients from currently running server."""
    for client_id in Client.instances:
        await core.sio.disconnect(client_id)


async def on_app_shutdown():
    """Disconnect clients, close running OAK device, and optionally start recording."""
    await disconnect()
    await close_camera()

    if getattr(app.state, "start_recording", False):
        subprocess_log(LOGS_PATH, "yolo_tracker_save_hqsync.py")

        with open(LOGS_PATH / "subprocess.log", "a", encoding="utf-8") as log_file_handle:
            subprocess.Popen(
                [sys.executable, str(BASE_PATH / "yolo_tracker_save_hqsync.py")],
                stdout=log_file_handle,
                stderr=log_file_handle,
                start_new_session=True
            )

    try:
        loop = asyncio.get_running_loop()
        loop.call_later(10, lambda: logger.warning("Web app forced to exit after timeout.") or sys.exit(0))
    except RuntimeError:
        logger.warning("No running event loop found, exiting immediately.")
        sys.exit(0)


def signal_handler(loop, signum, frame):
    """Handle a received signal to gracefully shut down the app."""
    logger.info("Signal received, initiating graceful app shutdown...")
    loop.create_task(close_camera())
    app.shutdown()


def on_app_startup(protocol, port, use_https):
    """Register signal handler and show startup message."""
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: signal_handler(loop, signal.SIGINT, None))
    loop.add_signal_handler(signal.SIGTERM, lambda: signal_handler(loop, signal.SIGTERM, None))

    logger.info("Insect Detect web app ready to go!")
    logger.info("Access via hostname:   %s://%s:%s", protocol, HOSTNAME, port)
    logger.info("Access via IP address: %s://%s:%s", protocol, IP_ADDRESS, port)
    if use_https:
        logger.info("Accept the self-signed SSL certificate in your browser when first connecting.")


if __name__ == "__main__":
    # Parse config and set parameters based on HTTPS setting
    config_selector = parse_yaml(BASE_PATH / "configs" / "config_selector.yaml")
    config_active = config_selector.config_active
    config = parse_yaml(BASE_PATH / "configs" / config_active)
    https_enabled = config.webapp.https.enabled
    ssl_cert_path = Path.home() / "ssl_certificates" / "cert.pem"
    ssl_key_path = Path.home() / "ssl_certificates" / "key.pem"
    use_https = https_enabled and ssl_cert_path.exists() and ssl_key_path.exists()
    if https_enabled and not use_https:
        logger.warning("HTTPS is enabled but no SSL certificates were found. Using HTTP instead.")
    protocol = "https" if use_https else "http"
    port = 8443 if use_https else 5000
    ssl_cert = str(ssl_cert_path) if use_https else None
    ssl_key = str(ssl_key_path) if use_https else None

    # Start the web app with specified parameters
    app.on_startup(lambda: on_app_startup(protocol, port, use_https))
    app.on_shutdown(on_app_shutdown)

    ui.run(
        host="0.0.0.0",
        port=port,
        title=f"{HOSTNAME} Web App",
        favicon=str(BASE_PATH / "static" / "favicon.ico"),
        binding_refresh_interval=0.2,  # refresh interval for active links (default: 0.1 seconds)
        show=False,
        reload=False,
        show_welcome_message=False,
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key
    )
