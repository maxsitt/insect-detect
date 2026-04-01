"""Stream OAK camera live feed and configure settings via browser-based web app.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Usage:
    Run with 'uv run webapp' from the insect-detect directory ('cd insect-detect').
    Configure settings via 'configs/config.yaml' (select active config in 'config_selector.yaml').

Web App Features:
    - Live MJPEG stream with SVG overlay showing bounding boxes and tracker/model metadata.
    - Interactive controls for zoom, manual focus and auto focus range preview.
    - Configuration sections for camera, detection, recording, processing, network and system settings.
    - Config file management: create, switch and save configuration files directly in the browser.
    - Deployment metadata input (location via browser Geolocation API, setting, distance, notes).
    - System info panel showing RPi and OAK chip temperature, CPU and RAM usage.
    - Log file viewer with color-coded output (errors, warnings, info).
    - Integrated browser terminal with full shell access to the RPi via pty.
    - Optional HTTPS support (required for browser Geolocation API).

Shutdown Behavior:
    - 'Stop App': shuts down the web app, optionally saves config beforehand.
    - 'Start Rec': shuts down the web app and launches the capture script as a new process.
"""

import asyncio
import base64
import copy
import logging
import os
import pty
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import cast

import depthai as dai
from fastapi.responses import StreamingResponse
from gpiozero import LED
from nicegui import Client, app, binding, core, run, ui
from nicegui.events import ValueChangeEventArguments, XtermDataEventArguments

from insectdetect.config import (AppConfig, check_config_changes, get_field_constraints,
                                 load_config_selector, load_config_yaml, update_config_selector,
                                 update_config_yaml)
from insectdetect.constants import (AUTO_RUN_MARKER, BASE_PATH, CONFIGS_PATH, HOSTNAME,
                                    LED_GPIO_PINS, LOGS_PATH, MODELS_PATH, RESOLUTION_PRESETS,
                                    STREAMING_MARKER, UV)
from insectdetect.metrics import configure_logger, get_oak_metrics, get_rpi_metrics, subprocess_log
from insectdetect.network import get_current_connection, get_ip_address, set_up_network
from insectdetect.oak import ZOOM_SIZES, create_pipeline, deletterbox_bbox

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Get IP address
IP_ADDRESS = get_ip_address()

# Create 1x1 black pixel PNG as placeholder image that will be shown when no frame is available
PLACEHOLDER_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
PLACEHOLDER_PNG_BYTES_LENGTH = str(len(PLACEHOLDER_PNG_BYTES)).encode()

# Field constraints extracted from AppConfig Pydantic model fields
# Each value is a dict with keys 'min', 'max' and 'multiple_of' (int/float or None)
_CONSTRAINT_PATHS: list[str] = [
    "camera.fps", "camera.zoom.factor", "camera.image.quality",
    "camera.focus.manual.distance", "camera.focus.manual.lens_pos",
    "camera.focus.range.distance.min", "camera.focus.range.distance.max",
    "camera.focus.range.lens_pos.min", "camera.focus.range.lens_pos.max",
    "camera.isp.sharpness", "camera.isp.luma_denoise", "camera.isp.chroma_denoise",
    "detection.num_shaves", "detection.conf_threshold",
    "recording.interval.detection", "recording.interval.timelapse", "recording.duration.default",
    "recording.duration.battery.high", "recording.duration.battery.medium", "recording.duration.battery.low",
    "webapp.stream.quality",
    "powermanager.charge_min", "powermanager.charge_check", "oak.temp_max", "oak.temp_check",
    "metrics.interval", "storage.disk_min", "storage.disk_check", "storage.archive.disk_low",
    "startup.auto_run.delay",
]

FIELD_CONSTRAINTS: dict[str, dict[str, int | float | None]] = {
    path: get_field_constraints(AppConfig, *path.split("."))
    for path in _CONSTRAINT_PATHS
}


@ui.page("/")
async def main_page() -> None:
    """Main entry point for the web app."""
    if AUTO_RUN_MARKER.exists() and not STREAMING_MARKER.exists():
        # Create marker file to indicate user interaction if in auto-run mode
        STREAMING_MARKER.touch()

    # Start camera if not already running
    if not getattr(app.state, "pipeline", None):
        await start_camera()

    # Create main UI content container (single column layout for responsive width and centering)
    with ui.column(align_items="center").classes("w-full max-w-3xl mx-auto"):
        create_ui_layout()

    # Create timer to get latest model/tracker data and update overlay (capped to 10 Hz)
    app.state.overlay_timer = ui.timer(max(app.state.refresh_interval, 0.1),
                                       update_overlay, immediate=False)

    # Create timer to update system information from RPi and OAK camera (capped to 0.2 Hz)
    app.state.sys_info_timer = ui.timer(5, update_sys_info, immediate=False)

    # Slow-blink LED to indicate web app is running
    if getattr(app.state, "config", None) and app.state.config.led.enabled:
        for _ in range(20):
            try:
                app.state.led = LED(app.state.config.led.gpio_pin)
                app.state.led.blink(on_time=1, off_time=1, background=True)
                break
            except Exception:
                await asyncio.sleep(0.1)


@ui.refreshable
def create_ui_layout() -> None:
    """Define layout for all UI elements."""
    create_video_stream_container()
    create_control_elements()

    with ui.card().tight().classes("w-full overflow-hidden border-l-4 border-lime-500"):
        with (ui.expansion("Deployment", icon="location_on").classes("w-full font-bold")
              .props('header-class="text-lime-500"')):
            create_deployment_section()

    with ui.card().tight().classes("w-full overflow-hidden border-l-4 border-emerald-500"):
        with (ui.expansion("Configuration", icon="settings").classes("w-full font-bold")
              .props('header-class="text-emerald-500"')):
            with ui.expansion("Camera Settings", icon="photo_camera").classes("w-full font-bold"):
                create_camera_settings()
            ui.separator().classes("bg-emerald-500 h-0.5")
            with ui.expansion("Detection Settings", icon="radar").classes("w-full font-bold"):
                create_detection_settings()
            ui.separator().classes("bg-emerald-500 h-0.5")
            with ui.expansion("Recording Settings", icon="videocam").classes("w-full font-bold"):
                create_recording_settings()
            ui.separator().classes("bg-emerald-500 h-0.5")
            with ui.expansion("Processing Settings", icon="tune").classes("w-full font-bold"):
                create_processing_settings()
            ui.separator().classes("bg-emerald-500 h-0.5")
            with ui.expansion("Web App Settings", icon="video_settings").classes("w-full font-bold"):
                create_webapp_settings()
            ui.separator().classes("bg-emerald-500 h-0.5")
            with ui.expansion("Network Settings", icon="network_wifi").classes("w-full font-bold"):
                create_network_settings()
            ui.separator().classes("bg-emerald-500 h-0.5")
            with ui.expansion("System Settings", icon="settings_applications").classes("w-full font-bold"):
                create_system_settings()
            ui.separator().classes("bg-emerald-500 h-0.5")
            with ui.expansion("Startup Settings", icon="rocket_launch").classes("w-full font-bold"):
                create_startup_settings()

    with ui.card().tight().classes("w-full overflow-hidden border-l-4 border-amber-300"):
        with (ui.expansion("Advanced", icon="build").classes("w-full font-bold")
              .props('header-class="text-amber-300"')):
            with ui.expansion("System Info", icon="monitor_heart").classes("w-full font-bold"):
                create_sys_info_section()
            ui.separator().classes("bg-amber-300 h-0.5")
            with ui.expansion("View Logs", icon="article").classes("w-full font-bold"):
                create_logs_section()
            ui.separator().classes("bg-amber-300 h-0.5")
            with ui.expansion("Terminal", icon="terminal").classes("w-full font-bold") as terminal_expansion:
                terminal_placeholder = ui.column().classes("w-full")

            terminal_initialized: list[bool] = [False]

            def on_terminal_expand(e: ValueChangeEventArguments) -> None:
                if e.value and not terminal_initialized[0]:
                    terminal_initialized[0] = True
                    with terminal_placeholder:
                        create_terminal_section()

            terminal_expansion.on_value_change(on_terminal_expand)

    with ui.row().classes("w-full justify-end mt-2 mb-4 gap-2"):
        (ui.button("Save Conf", on_click=save_config, color="green", icon="save")
         .props("dense"))
        (ui.button("Start Rec", on_click=start_recording, color="teal", icon="play_circle")
         .props("dense"))
        (ui.button("Stop App", on_click=confirm_shutdown, color="red", icon="power_settings_new")
         .props("dense"))


@app.get("/video/stream")
async def stream_mjpeg() -> StreamingResponse:
    """Stream MJPEG-encoded frames from OAK camera over HTTP."""
    return StreamingResponse(
        content=frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def create_video_stream_container() -> None:
    """Create video stream container with fixed aspect ratio and row with camera parameters."""
    w = app.state.frame_size[0]
    h = app.state.frame_size[1]
    with ui.element("div").classes("w-full p-0 overflow-hidden bg-black border border-gray-700"):
        with ui.element("div").classes(f"w-full aspect-[{w}/{h}]"):
            with ui.element("div").classes("w-full h-full relative"):
                app.state.frame_ii = (ui.interactive_image(source="/video/stream", sanitize=False)
                                      .classes("w-full h-full object-contain"))
                # Add separate SVG layer for ROI overlay
                app.state.roi_layer = app.state.frame_ii.add_layer()

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


async def start_camera() -> None:
    """Upload pipeline to OAK device and start camera with selected configuration."""
    config_selector = load_config_selector()
    app.state.config_active = config_selector.config_active
    app.state.config = load_config_yaml(CONFIGS_PATH / app.state.config_active)
    app.state.config_updates = copy.deepcopy(app.state.config.model_dump())
    app.state.models = sorted([f.name for f in MODELS_PATH.glob("*.tar.xz")])
    app.state.logs = sorted([f.name for f in LOGS_PATH.glob("*.log")])
    app.state.configs = sorted([f.name for f in CONFIGS_PATH.glob("*.yaml")
                                if f.name != "config_selector.yaml"])
    app.state.connection = get_current_connection()
    app.state.refresh_interval = max(round(1 / app.state.config.camera.fps, 3), 0.05)
    app.state.show_overlay = True
    app.state.last_overlay_empty = True
    app.state.last_roi_layer_empty = True
    app.state.ae_region_active = False
    app.state.last_ae_time = 0.0
    app.state.start_recording = False
    app.state.focus_initialized = False
    app.state.focus_mode_initialized = False
    app.state.manual_focus_enabled = app.state.config.camera.focus.mode == "manual"
    app.state.focus_range_enabled = app.state.config.camera.focus.mode == "range"
    app.state.focus_distance_enabled = app.state.config.camera.focus.type == "distance"
    app.state.frame_count = 0
    app.state.fps = 0
    app.state.lens_pos = 0
    app.state.iso_sens = 0
    app.state.exp_time = 0
    app.state.prev_time = time.monotonic()
    app.state.sys_info = {
        "rpi_cpu_temp": "NA",
        "rpi_cpu_usage_avg": "NA",
        "rpi_cpu_usage_sum": "NA",
        "rpi_ram_usage": "NA",
        "rpi_ram_available": "NA",
        "oak_chip_temp": "NA",
        "oak_cpu_usage_css": "NA",
        "oak_cpu_usage_mss": "NA",
        "oak_ram_usage_ddr": "NA",
        "oak_ram_available_ddr": "NA",
        "oak_ram_usage_css": "NA",
        "oak_ram_available_css": "NA",
        "oak_ram_usage_mss": "NA",
        "oak_ram_available_mss": "NA",
        "oak_ram_usage_cmx": "NA",
        "oak_ram_available_cmx": "NA",
    }

    (app.state.pipeline, app.state.q_frames, app.state.q_tracks, app.state.q_syslog,
     app.state.q_camctrl, app.state.frame_size, app.state.nn_input_size, app.state.sensor_roi,
     app.state.labels) = create_pipeline(app.state.config, stream=True)

    app.state.pipeline.start()

    # Ensure camera starts in continuous auto focus mode if manual focus is not enabled
    # Workaround to avoid focus issues if setAutoFocusRegion() was used in initial camera control
    if app.state.config.camera.focus.mode != "manual":
        af_ctrl = dai.CameraControl().setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        app.state.q_camctrl.send(af_ctrl)

    ui.notification("OAK camera pipeline started!", type="positive", timeout=2)


async def close_camera() -> None:
    """Stop streaming and OAK pipeline, clean up resources."""
    for queue in ("q_frames", "q_tracks", "q_syslog", "q_camctrl"):
        if getattr(app.state, queue, None):
            setattr(app.state, queue, None)

    if getattr(app.state, "overlay_timer", None):
        app.state.overlay_timer.deactivate()
        app.state.overlay_timer = None

    if getattr(app.state, "sys_info_timer", None):
        app.state.sys_info_timer.deactivate()
        app.state.sys_info_timer = None

    if getattr(app.state, "pipeline", None):
        app.state.pipeline.stop()
        app.state.pipeline = None

    if getattr(app.state, "led", None):
        app.state.led.close()
        app.state.led = None


def get_frame(q_frames: dai.MessageQueue | None) -> tuple[bytes, bytes, int, int, float] | None:
    """Get MJPEG-encoded frame and associated metadata from the OAK camera output queue."""
    if not q_frames:
        return None
    try:
        frame = cast(dai.ImgFrame | None, q_frames.tryGet())
        if frame is None:
            return None
        frame_bytes: bytes = frame.getData().tobytes()
        frame_bytes_length: bytes = str(len(frame_bytes)).encode()
        lens_pos: int = frame.getLensPosition()
        iso_sens: int = frame.getSensitivity()
        exp_time: float = frame.getExposureTime().total_seconds() * 1000
        return frame_bytes, frame_bytes_length, lens_pos, iso_sens, exp_time
    except Exception:
        return None


async def frame_generator() -> AsyncGenerator[bytes, None]:
    """Yield MJPEG-encoded frames asynchronously and update camera parameters."""
    try:
        next_tick = time.monotonic()
        while getattr(app.state, "q_frames", None):
            frame_data = get_frame(app.state.q_frames)
            if frame_data is not None:
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


async def get_tracker_data() -> list[dict] | None:
    """Get model/tracker data from the OAK camera output queue, set AE region if enabled."""
    tracker_data: list[dict] = []
    if getattr(app.state, "q_tracks", None):
        track = cast(dai.Tracklets | None, app.state.q_tracks.tryGet())
        if track is None:
            return None

        tracklet_id_max = -1
        bbox_max: tuple[float, float, float, float] | None = None

        for tracklet in track.tracklets:
            # Only process active tracklets (not "LOST" or "REMOVED")
            tracklet_status = tracklet.status.name
            if tracklet_status in {"TRACKED", "NEW"}:
                tracklet_id: int = tracklet.id
                bbox_raw: tuple[float, float, float, float] = (
                    max(0.0, min(1.0, tracklet.srcImgDetection.xmin)),
                    max(0.0, min(1.0, tracklet.srcImgDetection.ymin)),
                    max(0.0, min(1.0, tracklet.srcImgDetection.xmax)),
                    max(0.0, min(1.0, tracklet.srcImgDetection.ymax))
                )
                # De-letterbox bbox from NN-normalized space to frame-normalized space
                bbox = deletterbox_bbox(
                    bbox_raw,
                    app.state.frame_size[0], app.state.frame_size[1],
                    app.state.nn_input_size[0], app.state.nn_input_size[1]
                )

                tracklet_data = {
                    "label": app.state.labels[tracklet.srcImgDetection.label],
                    "confidence": round(tracklet.srcImgDetection.confidence, 2),
                    "track_id": tracklet_id,
                    "track_status": tracklet_status,
                    "x_min": round(bbox[0], 4),
                    "y_min": round(bbox[1], 4),
                    "x_max": round(bbox[2], 4),
                    "y_max": round(bbox[3], 4)
                }
                tracker_data.append(tracklet_data)

                # Track most recent active tracking ID and its bounding box
                if tracklet_status == "TRACKED" and tracklet_id > tracklet_id_max:
                    tracklet_id_max = tracklet_id
                    bbox_max = bbox

        if app.state.config_updates["detection"]["ae_region"]["enabled"]:
            if bbox_max:
                ae_time = time.monotonic()
                if ae_time - app.state.last_ae_time >= 1.0:
                    # Set AE region to bbox of most recent active tracking ID (capped to 1 Hz)
                    # Map bbox (frame-normalized) to sensor-space coordinates
                    roi_x, roi_y, roi_w, roi_h = app.state.sensor_roi
                    rect_bbox: tuple[int, int, int, int] = (
                        max(1, round(roi_x + bbox_max[0] * roi_w)),
                        max(1, round(roi_y + bbox_max[1] * roi_h)),
                        max(10, round((bbox_max[2] - bbox_max[0]) * roi_w)),
                        max(10, round((bbox_max[3] - bbox_max[1]) * roi_h)),
                    )
                    exp_ctrl = dai.CameraControl().setAutoExposureRegion(*rect_bbox)
                    app.state.q_camctrl.send(exp_ctrl)
                    app.state.ae_region_active = True
                    app.state.last_ae_time = ae_time
            elif app.state.ae_region_active:
                # Reset AE region to full visible FOV in sensor space if no active tracking ID
                exp_ctrl = dai.CameraControl().setAutoExposureRegion(*app.state.sensor_roi)
                app.state.q_camctrl.send(exp_ctrl)
                app.state.ae_region_active = False
                app.state.last_ae_time = 0.0

    return tracker_data


def build_tracker_overlay(tracker_data: list[dict]) -> str:
    """Build SVG content with latest model/tracker data.

    Bounding box coordinates in frame-normalized space are remapped to
    the SVG viewBox="0 0 1 1" coordinate system via aspect ratio transform.

    Args:
        tracker_data: List of tracklet data dicts.

    Returns:
        SVG string with bounding boxes and label/track ID text for all active tracklets.
    """
    aspect_ratio = app.state.frame_size[0] / app.state.frame_size[1]

    svg_parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1" width="100%" height="100%" '
        'style="position:absolute; top:0; left:0; pointer-events:none;">'
    ]

    for data in tracker_data:
        label = data["label"]
        confidence = data["confidence"]
        tracklet_id = data["track_id"]
        x_min = (data["x_min"] - 0.5) * aspect_ratio + 0.5
        x_max = (data["x_max"] - 0.5) * aspect_ratio + 0.5
        y_min = data["y_min"]
        width  = x_max - x_min
        height = data["y_max"] - y_min

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
            f'<tspan x="{x_min}" dy="0.04">ID: {tracklet_id}</tspan></text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def build_roi_overlay(zoom: float) -> str:
    """Build SVG content showing a center-crop rectangle to visualize the pending zoom factor.

    Shown when the configured zoom factor is higher than the currently active pipeline zoom,
    as a preview of the new crop region on the currently streamed frame.

    Args:
        zoom: Configured (pending) zoom factor to preview.

    Returns:
        SVG string with the ROI boundary rectangle.
    """
    # Pending crop dimensions from precomputed lookup table (exact aligned pixel values)
    pending_w, pending_h = ZOOM_SIZES[app.state.config.camera.image.resolution]["stream"][zoom]

    # Center the pending crop within the active frame
    x_min_norm = (app.state.frame_size[0] - pending_w) / 2 / app.state.frame_size[0]
    y_min_norm = (app.state.frame_size[1] - pending_h) / 2 / app.state.frame_size[1]
    x_max_norm = 1.0 - x_min_norm
    y_max_norm = 1.0 - y_min_norm

    # Remap to SVG viewBox="0 0 1 1" with aspect ratio transform
    aspect_ratio = app.state.frame_size[0] / app.state.frame_size[1]
    x_min = (x_min_norm - 0.5) * aspect_ratio + 0.5
    x_max = (x_max_norm - 0.5) * aspect_ratio + 0.5
    y_min = y_min_norm
    width  = x_max - x_min
    height = y_max_norm - y_min

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1" width="100%" height="100%" '
        'style="position:absolute; top:0; left:0; pointer-events:none;">'
        # Black outer stroke for visibility on bright backgrounds
        f'<rect x="{x_min}" y="{y_min}" width="{width}" height="{height}" '
        'fill="none" stroke="black" stroke-width="0.008" stroke-opacity="0.6"/>'
        # White inner stroke for visibility on dark backgrounds
        f'<rect x="{x_min}" y="{y_min}" width="{width}" height="{height}" '
        'fill="none" stroke="white" stroke-width="0.004" stroke-opacity="0.9"/>'
        '</svg>'
    )


async def update_overlay() -> None:
    """Update SVG overlay with pending zoom ROI (if applicable) and latest model/tracker data."""
    if getattr(app.state, "roi_layer", None) is not None:
        zoom_enabled: bool = app.state.config_updates["camera"]["zoom"]["enabled"]
        zoom_factor: float = app.state.config_updates["camera"]["zoom"]["factor"]
        active_zoom_factor: float = (
            app.state.config.camera.zoom.factor
            if app.state.config.camera.zoom.enabled
            else 1.0
        )
        if zoom_enabled and zoom_factor > active_zoom_factor:
            app.state.roi_layer.content = build_roi_overlay(zoom_factor)
            app.state.last_roi_layer_empty = False
        elif not getattr(app.state, "last_roi_layer_empty", False):
            app.state.roi_layer.content = ""
            app.state.last_roi_layer_empty = True

    if app.state.show_overlay or app.state.config_updates["detection"]["ae_region"]["enabled"]:
        tracker_data = await get_tracker_data()
        if not tracker_data:
            if not getattr(app.state, "last_overlay_empty", False):
                app.state.frame_ii.content = ""
                app.state.last_overlay_empty = True
            return
        if app.state.show_overlay:
            app.state.frame_ii.content = build_tracker_overlay(tracker_data)
            app.state.last_overlay_empty = False
    else:
        if not getattr(app.state, "last_overlay_empty", False):
            app.state.frame_ii.content = ""
            app.state.last_overlay_empty = True


def reset_zoom() -> None:
    """Reset zoom factor to 1.0 and show notification to save config and reload."""
    app.state.config_updates["camera"]["zoom"]["factor"] = 1.0
    ui.notification(
        "Zoom reset to 1.0x. Save config and reload the web app to apply.",
        type="warning", timeout=4
    )


async def set_manual_focus(e: ValueChangeEventArguments) -> None:
    """Set manual focus position of OAK camera."""
    if not app.state.focus_initialized:
        app.state.focus_initialized = True
        return
    if not getattr(app.state, "q_camctrl", None):
        return
    mf_ctrl = dai.CameraControl()
    mf_ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
    mf_ctrl.setManualFocus(e.value)
    app.state.q_camctrl.send(mf_ctrl)


async def preview_focus_range(e: ValueChangeEventArguments) -> None:
    """Set manual focus position of OAK camera to the last changed focus range position."""
    if not app.state.focus_initialized or not hasattr(app.state, "previous_lens_pos_range"):
        app.state.focus_initialized = True
        app.state.previous_lens_pos_range = e.value
        return
    if not getattr(app.state, "q_camctrl", None):
        app.state.previous_lens_pos_range = e.value
        return
    if app.state.previous_lens_pos_range["min"] != e.value["min"]:
        mf_ctrl = dai.CameraControl()
        mf_ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        mf_ctrl.setManualFocus(e.value["min"])
        app.state.q_camctrl.send(mf_ctrl)
    elif app.state.previous_lens_pos_range["max"] != e.value["max"]:
        mf_ctrl = dai.CameraControl()
        mf_ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        mf_ctrl.setManualFocus(e.value["max"])
        app.state.q_camctrl.send(mf_ctrl)
    app.state.previous_lens_pos_range = e.value


def refresh_config_select_ui() -> None:
    """Refresh config selector UI to reflect any files added or deleted since last load."""
    app.state.configs = sorted([f.name for f in CONFIGS_PATH.glob("*.yaml")
                                if f.name != "config_selector.yaml"])
    app.state.config_select_ui.set_options(app.state.configs, value=app.state.config_active)


async def on_config_change(e: ValueChangeEventArguments) -> None:
    """Switch to selected config file and apply new configuration parameters."""
    config_selected_name = e.value
    if config_selected_name == app.state.config_active:
        return

    # Check if the config file still exists (may have been deleted manually)
    config_selected_path = CONFIGS_PATH / config_selected_name
    if not config_selected_path.exists():
        ui.notification(
            f"Config file '{config_selected_name}' no longer exists and has been removed.",
            type="warning", timeout=4
        )
        refresh_config_select_ui()
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
        if action == "save":
            app.state.config_select_ui.set_value(app.state.config_active)
            await save_config()
            return

    config_selected = load_config_yaml(CONFIGS_PATH / config_selected_name)
    has_network_changes = check_config_changes(
        app.state.config.network.model_dump(),
        config_selected.network.model_dump()
    )

    await apply_config_changes(config_selected_name, has_network_changes, config_selected)


def create_control_elements() -> None:
    """Create UI elements and config binding for camera, web app and config control."""
    # Slider for zoom factor visualized by center-crop ROI overlay (only visible if zoom is enabled)
    with (ui.column().classes("w-full gap-0 mb-0")
          .bind_visibility_from(app.state.config_updates["camera"]["zoom"], "enabled")):
        ui.label("Zoom:").classes("font-bold")
        c = FIELD_CONSTRAINTS["camera.zoom.factor"]
        assert c["min"] is not None and c["max"] is not None
        active_zoom_factor: float = (
            app.state.config.camera.zoom.factor
            if app.state.config.camera.zoom.enabled
            else float(c["min"])
        )
        slider_min = max(float(c["min"]), active_zoom_factor)
        with ui.row(align_items="center").classes("w-full gap-4"):
            (ui.slider(min=slider_min, max=float(c["max"]), step=0.1)
             .props("label snap")
             .classes("flex-1")
             .bind_value(app.state.config_updates["camera"]["zoom"], "factor",
                         forward=lambda v: round(float(v), 1) if v is not None else 1.0))
            (ui.button(icon="zoom_out", color="orange", on_click=reset_zoom)
             .props("dense round")
             .tooltip("Reset zoom to 1.0x"))

    # Slider for manual focus control (only visible if focus mode is set to "manual")
    with ui.column().classes("w-full gap-0 mb-0").bind_visibility_from(app.state, "manual_focus_enabled"):
        ui.label("Manual Focus:").classes("font-bold")
        c = FIELD_CONSTRAINTS["camera.focus.manual.lens_pos"]
        assert c["min"] is not None and c["max"] is not None
        (ui.slider(min=float(c["min"]), max=float(c["max"]), step=1, on_change=set_manual_focus)
         .props("label")
         .bind_value(app.state.config_updates["camera"]["focus"]["manual"], "lens_pos",
                     forward=lambda v: int(v) if v is not None else None))

    # Slider for auto focus range control (only visible if focus mode is set to "range")
    with ui.column().classes("w-full gap-0 mb-0").bind_visibility_from(app.state, "focus_range_enabled"):
        ui.label("Focus Range:").classes("font-bold")
        c_min = FIELD_CONSTRAINTS["camera.focus.range.lens_pos.min"]
        c_max = FIELD_CONSTRAINTS["camera.focus.range.lens_pos.max"]
        assert c_min["min"] is not None and c_max["max"] is not None
        (ui.range(min=float(c_min["min"]), max=float(c_max["max"]), step=1, on_change=preview_focus_range)
         .props("label")
         .bind_value(app.state.config_updates["camera"]["focus"]["range"], "lens_pos",
                     forward=lambda v: {"min": int(v["min"]), "max": int(v["max"])} if v is not None else None))

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
        app.state.config_select_ui = (
            ui.select(app.state.configs, value=app.state.config_active, on_change=on_config_change)
            .classes("flex-1 truncate")
            .on("click", refresh_config_select_ui)
        )


async def get_location() -> None:
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
            return

        app.state.config_updates["deployment"]["location"]["latitude"] = response["latitude"]
        app.state.config_updates["deployment"]["location"]["longitude"] = response["longitude"]
        if "accuracy" in response:
            app.state.config_updates["deployment"]["location"]["accuracy"] = round(response["accuracy"], 1)

    except TimeoutError:
        ui.notification("Location request timed out. Please try again", type="warning", timeout=3)
        return


def grid_separator() -> None:
    """Create a horizontal separator line for a 2-column grid layout."""
    with ui.row().classes("w-full col-span-2 py-0 my-0"):
        ui.element("div").classes("w-full border-t border-gray-700")


def validate_number(
    value: float | None,
    min_value: int | float | None,
    max_value: int | float | None,
    multiple: int | float | None = None
) -> bool:
    """Validate that a number is within the required range and optionally a multiple.

    Args:
        value:     Value to validate (None is treated as invalid).
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).
        multiple:  If provided, value must be a multiple of this number.

    Returns:
        True if the value is valid, False otherwise.
    """
    if value is None or min_value is None or max_value is None:
        return False
    if not min_value <= value <= max_value:
        return False
    if multiple is not None:
        return value % multiple == 0
    return True


def create_deployment_section() -> None:
    """Create UI elements and config binding for deployment metadata."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        (ui.label("Start Time").classes("font-bold")
         .tooltip("Start time of the camera deployment (ISO 8601 format)"))
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
                (ui.number(label="Accuracy", min=0, max=500, precision=1, step=1, suffix="m")
                 .bind_value(app.state.config_updates["deployment"]["location"], "accuracy",
                             forward=lambda v: float(v) if v not in (None, "") else None))
            loc_button = ui.button("Get Location", icon="my_location", on_click=get_location)
            if not app.state.config.webapp.https.enabled:
                loc_button.disable()
                loc_button.tooltip("HTTPS must be enabled to use the browser Geolocation API")

        grid_separator()
        (ui.label("Setting").classes("font-bold")
         .tooltip("Background setting of the camera (e.g. platform type/flower species)"))
        (ui.input(placeholder="Enter background setting").props("clearable")
         .bind_value(app.state.config_updates["deployment"], "setting",
                     forward=lambda v: str(v) if v not in (None, "") else None))

        grid_separator()
        (ui.label("Distance").classes("font-bold")
         .tooltip("Distance from the camera to the background (e.g. platform/flower)"))
        (ui.number(label="Distance", min=8, max=100, precision=0, step=1, suffix="cm",
                   validation={"Optional value between 8-100":
                               lambda v: v in (None, "") or validate_number(v, 8, 100)})
         .bind_value(app.state.config_updates["deployment"], "distance",
                     forward=lambda v: int(v) if v not in (None, "") else None))

        grid_separator()
        (ui.label("Notes").classes("font-bold")
         .tooltip("Additional fieldnotes about the deployment"))
        (ui.textarea(placeholder="Enter deployment notes").props("clearable")
         .bind_value(app.state.config_updates["deployment"], "notes",
                     forward=lambda v: str(v) if v not in (None, "") else None))


async def on_focus_mode_change(e: ValueChangeEventArguments) -> None:
    """Update relevant focus parameters in config, set continuous focus if selected."""
    if not app.state.focus_mode_initialized:
        app.state.focus_mode_initialized = True
        return
    app.state.manual_focus_enabled = e.value == "manual"
    app.state.focus_range_enabled = e.value == "range"
    if e.value == "continuous":
        if not getattr(app.state, "q_camctrl", None):
            return
        af_ctrl = dai.CameraControl().setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        app.state.q_camctrl.send(af_ctrl)
    else:
        app.state.focus_initialized = False
        app.state.focus_distance_enabled = False
        app.state.config_updates["camera"]["focus"]["type"] = "lens_pos"
        ui.notification("Focus type set to 'lens_pos' for manual adjustment with the control slider.",
                        type="warning", timeout=3)


async def on_focus_type_change(e: ValueChangeEventArguments) -> None:
    """Update focus distance visibility when focus type changes."""
    app.state.focus_distance_enabled = e.value == "distance"
    if e.value == "distance":
        ui.notification("Focus control slider will still use lens position for finer adjustment.",
                        type="warning", timeout=3)


def create_camera_settings() -> None:
    """Create UI elements and config binding for camera settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        ui.label("Focus Mode").classes("font-bold")
        (ui.select(["continuous", "manual", "range"], label="Mode", on_change=on_focus_mode_change)
         .bind_value(app.state.config_updates["camera"]["focus"], "mode"))

        grid_separator()
        ui.label("Focus Type").classes("font-bold")
        with ui.column().classes("w-full gap-1"):
            (ui.select(["lens_pos", "distance"], label="Type", on_change=on_focus_type_change)
             .classes("w-full")
             .bind_value(app.state.config_updates["camera"]["focus"], "type"))

            with (ui.column().classes("w-full gap-1")
                  .bind_visibility_from(app.state, "focus_distance_enabled")):
                c = FIELD_CONSTRAINTS["camera.focus.manual.distance"]
                with ui.row(align_items="center").classes("w-full gap-2"):
                    (ui.number(label="Manual Focus",
                               placeholder=app.state.config.camera.focus.manual.distance,
                               min=c["min"], max=c["max"], precision=0, step=1, suffix="cm",
                               validation={f"Required value between {c['min']}-{c['max']}":
                                           lambda v, c=c: validate_number(v, c["min"], c["max"])}).classes("flex-1")
                     .bind_value(app.state.config_updates["camera"]["focus"]["manual"], "distance",
                                 forward=lambda v: int(v) if v is not None else None))
                c_min = FIELD_CONSTRAINTS["camera.focus.range.distance.min"]
                c_max = FIELD_CONSTRAINTS["camera.focus.range.distance.max"]
                with ui.row(align_items="center").classes("w-full gap-2"):
                    (ui.number(label="Range Min",
                               placeholder=app.state.config.camera.focus.range.distance.min,
                               min=c_min["min"], max=c_min["max"], precision=0, step=1, suffix="cm",
                               validation={f"Required value between {c_min['min']}-{c_min['max']}":
                                           lambda v, c=c_min: validate_number(v, c["min"], c["max"])}).classes("flex-1")
                     .bind_value(app.state.config_updates["camera"]["focus"]["range"]["distance"], "min",
                                 forward=lambda v: int(v) if v is not None else None))
                    (ui.number(label="Range Max",
                               placeholder=app.state.config.camera.focus.range.distance.max,
                               min=c_max["min"], max=c_max["max"], precision=0, step=1, suffix="cm",
                               validation={f"Required value between {c_max['min']}-{c_max['max']}":
                                           lambda v, c=c_max: validate_number(v, c["min"], c["max"])}).classes("flex-1")
                     .bind_value(app.state.config_updates["camera"]["focus"]["range"]["distance"], "max",
                                 forward=lambda v: int(v) if v is not None else None))
                (ui.label("Focus control slider will still use lens position for finer adjustment!")
                 .classes("text-xs text-gray-500"))

        grid_separator()
        (ui.label("Center-crop Zoom").classes("font-bold")
         .tooltip("Crop the field of view around the center of the frame. "
                  "Applied to both saved images and the detection model input. "
                  "Save config and restart web app to take effect."))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["camera"]["zoom"], "enabled"))
            c = FIELD_CONSTRAINTS["camera.zoom.factor"]
            assert c["min"] is not None and c["max"] is not None
            (ui.number(label="Zoom Factor", placeholder=app.state.config.camera.zoom.factor,
                       min=float(c["min"]), max=float(c["max"]), precision=1, step=0.1, suffix="x",
                       validation={f"Required value between {c['min']}-{c['max']}":
                                   lambda v, c=c: validate_number(v, c["min"], c["max"])}).classes("w-full")
             .bind_visibility_from(app.state.config_updates["camera"]["zoom"], "enabled")
             .bind_value(app.state.config_updates["camera"]["zoom"], "factor",
                         forward=lambda v: round(float(v), 1) if v is not None else 1.0))

        grid_separator()
        c = FIELD_CONSTRAINTS["camera.fps"]
        ui.label("Frame Rate").classes("font-bold").tooltip("Higher FPS increases power consumption")
        (ui.number(label="FPS", placeholder=app.state.config.camera.fps,
                   min=c["min"], max=c["max"], precision=0, step=1,
                   validation={f"Required value between {c['min']}-{c['max']}":
                               lambda v, c=c: validate_number(v, c["min"], c["max"])})
         .bind_value(app.state.config_updates["camera"], "fps",
                     forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("Resolution").classes("font-bold")
         .tooltip("Resolution preset for captured images and stream. "
                  "Save config and restart to take effect.\n"
                  "4k: 3840x2160 | 4k-square: 2176x2160 | 1080p: 1920x1080 | 1080p-square: 1088x1080"))
        (ui.select(list(RESOLUTION_PRESETS.keys()), label="Preset")
         .bind_value(app.state.config_updates["camera"]["image"], "resolution"))

        grid_separator()
        c = FIELD_CONSTRAINTS["camera.image.quality"]
        ui.label("JPEG Quality").classes("font-bold").tooltip("JPEG quality of captured images")
        (ui.number(label="JPEG", placeholder=app.state.config.camera.image.quality,
                   min=c["min"], max=c["max"], precision=0, step=1,
                   validation={f"Required value between {c['min']}-{c['max']}":
                               lambda v, c=c: validate_number(v, c["min"], c["max"])})
         .bind_value(app.state.config_updates["camera"]["image"], "quality",
                     forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("ISP Settings").classes("font-bold")
         .tooltip("Setting Sharpness and Luma Denoise to 0 can reduce artifacts"))
        with ui.row(align_items="center").classes("w-full gap-2"):
            c = FIELD_CONSTRAINTS["camera.isp.sharpness"]
            (ui.number(label="Sharpness", placeholder=app.state.config.camera.isp.sharpness,
                       min=c["min"], max=c["max"], precision=0, step=1,
                       validation={f"Required value between {c['min']}-{c['max']}":
                                   lambda v, c=c: validate_number(v, c["min"], c["max"])})
             .classes("flex-1")
             .bind_value(app.state.config_updates["camera"]["isp"], "sharpness",
                         forward=lambda v: int(v) if v is not None else None))
            c = FIELD_CONSTRAINTS["camera.isp.luma_denoise"]
            (ui.number(label="Luma Denoise", placeholder=app.state.config.camera.isp.luma_denoise,
                       min=c["min"], max=c["max"], precision=0, step=1,
                       validation={f"Required value between {c['min']}-{c['max']}":
                                   lambda v, c=c: validate_number(v, c["min"], c["max"])})
             .classes("flex-1")
             .bind_value(app.state.config_updates["camera"]["isp"], "luma_denoise",
                         forward=lambda v: int(v) if v is not None else None))
            c = FIELD_CONSTRAINTS["camera.isp.chroma_denoise"]
            (ui.number(label="Chroma Denoise", placeholder=app.state.config.camera.isp.chroma_denoise,
                       min=c["min"], max=c["max"], precision=0, step=1,
                       validation={f"Required value between {c['min']}-{c['max']}":
                                   lambda v, c=c: validate_number(v, c["min"], c["max"])})
             .classes("flex-1")
             .bind_value(app.state.config_updates["camera"]["isp"], "chroma_denoise",
                         forward=lambda v: int(v) if v is not None else None))


async def on_ae_region_change(e: ValueChangeEventArguments) -> None:
    """Reset auto AE region to full frame if setting is disabled."""
    if not e.value and app.state.ae_region_active:
        if not getattr(app.state, "q_camctrl", None):
            return
        exp_ctrl = dai.CameraControl().setAutoExposureRegion(*app.state.sensor_roi)
        app.state.q_camctrl.send(exp_ctrl)
        app.state.ae_region_active = False


def create_detection_settings() -> None:
    """Create UI elements and config binding for detection settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        ui.label("Detection Model").classes("font-bold")
        (ui.select(app.state.models, label="Model", value=app.state.config.detection.model)
         .classes("truncate")
         .bind_value(app.state.config_updates["detection"], "model"))

        grid_separator()
        c = FIELD_CONSTRAINTS["detection.num_shaves"]
        (ui.label("Number of SHAVEs").classes("font-bold")
         .tooltip("Number of SHAVEs (compute cores) the compiled model can use"))
        (ui.number(label="SHAVEs", placeholder=app.state.config.detection.num_shaves,
                   min=c["min"], max=c["max"], precision=0, step=1,
                   validation={f"Required value between {c['min']}-{c['max']}":
                               lambda v, c=c: validate_number(v, c["min"], c["max"])})
         .bind_value(app.state.config_updates["detection"], "num_shaves",
                     forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        c = FIELD_CONSTRAINTS["detection.conf_threshold"]
        ui.label("Confidence Threshold").classes("font-bold").tooltip("Overrides model config file")
        (ui.number(label="Confidence", placeholder=app.state.config.detection.conf_threshold,
                   min=c["min"], max=c["max"], precision=2, step=0.01,
                   validation={f"Required value between {c['min']}-{c['max']}":
                               lambda v, c=c: validate_number(v, c["min"], c["max"])})
         .bind_value(app.state.config_updates["detection"], "conf_threshold",
                     forward=lambda v: float(v) if v is not None else None))

        grid_separator()
        (ui.label("Detection-based Exposure").classes("font-bold")
         .tooltip("Use bounding box coordinates from detections to set auto exposure region"))
        (ui.switch("Enable", on_change=on_ae_region_change).props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["detection"]["ae_region"], "enabled"))


def create_duration_input(
    duration_type: str,
    label_text: str,
    tooltip_text: str | None = None
) -> None:
    """Create a minutes input field for a specific duration type."""
    if duration_type == "default":
        config_target = app.state.config_updates["recording"]["duration"]
        target_key = "default"
        c = FIELD_CONSTRAINTS["recording.duration.default"]
    else:
        config_target = app.state.config_updates["recording"]["duration"]["battery"]
        target_key = duration_type
        c = FIELD_CONSTRAINTS[f"recording.duration.battery.{duration_type}"]

    label = ui.label(label_text).classes("font-bold")
    if tooltip_text:
        label.tooltip(tooltip_text)

    (ui.number(label="Duration", placeholder=config_target[target_key],
               min=c["min"], max=c["max"], precision=0, step=1, suffix="minutes",
               validation={f"Required value between {c['min']}-{c['max']}":
                           lambda v, c=c: validate_number(v, c["min"], c["max"])})
     .bind_value(config_target, target_key,
                 forward=lambda v: int(v) if v is not None else None))


def create_recording_settings() -> None:
    """Create UI elements and config binding for recording settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        ui.label("Duration").classes("font-bold").tooltip("Duration per recording session")
        with ui.column().classes("w-full"):
            with ui.tabs().classes("w-full") as tabs:
                ui.tab("Battery", icon="battery_charging_full")
                ui.tab("No Battery", icon="timer")
            with ui.tab_panels(tabs, value="Battery").classes("w-full"):
                with ui.tab_panel("Battery"):
                    create_duration_input("high", "High (> 70% or USB connected)",
                        "Duration if battery charge level is > 70% or USB power is connected")
                    create_duration_input("medium", "Medium (50-70%)",
                        "Duration if battery charge level is between 50-70%")
                    create_duration_input("low", "Low (30-50%)",
                        "Duration if battery charge level is between 30-50%")
                with ui.tab_panel("No Battery"):
                    create_duration_input("default", "Default",
                        "Duration if powermanager is disabled")

        grid_separator()
        ui.label("Capture Interval").classes("font-bold")
        with ui.column().classes("w-full"):
            with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):
                c = FIELD_CONSTRAINTS["recording.interval.detection"]
                (ui.label("Detection").classes("font-bold")
                 .tooltip("Interval for saving full frame + metadata while object is detected"))
                (ui.number(label="Capture Interval",
                           placeholder=app.state.config.recording.interval.detection,
                           min=c["min"], max=c["max"], precision=1, step=0.1, suffix="seconds",
                           validation={f"Required value between {c['min']}-{c['max']}":
                                       lambda v, c=c: validate_number(v, c["min"], c["max"])})
                 .bind_value(app.state.config_updates["recording"]["interval"], "detection",
                             forward=lambda v: float(v) if v is not None else None))
                c = FIELD_CONSTRAINTS["recording.interval.timelapse"]
                (ui.label("Timelapse").classes("font-bold")
                 .tooltip("Interval for saving full frame (independent of detected objects)"))
                (ui.number(label="Capture Interval",
                           placeholder=app.state.config.recording.interval.timelapse,
                           min=c["min"], max=c["max"], precision=1, step=0.1, suffix="seconds",
                           validation={f"Required value between {c['min']}-{c['max']}":
                                       lambda v, c=c: validate_number(v, c["min"], c["max"])})
                 .bind_value(app.state.config_updates["recording"]["interval"], "timelapse",
                             forward=lambda v: float(v) if v is not None else None))

        grid_separator()
        (ui.label("Shutdown After Recording").classes("font-bold")
         .tooltip("Shut down Raspberry Pi after recording session is finished or interrupted"))
        (ui.switch("Enable").props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["recording"]["shutdown"], "enabled"))


def create_processing_settings() -> None:
    """Create UI elements and config binding for processing settings."""
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        (ui.label("Crop Detections").classes("font-bold")
         .tooltip("Crop individual detections from full images and save as separate files"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["processing"]["crop"], "enabled"))
            (ui.select(["square", "original"], label="Crop Method").classes("w-full")
             .bind_visibility_from(app.state.config_updates["processing"]["crop"], "enabled")
             .bind_value(app.state.config_updates["processing"]["crop"], "method"))

        grid_separator()
        (ui.label("Draw Overlays").classes("font-bold")
         .tooltip("Draw bounding boxes and metadata overlays on full images and save as copies"))
        (ui.switch("Enable").props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["processing"]["overlay"], "enabled"))

        grid_separator()
        (ui.label("Delete Originals").classes("font-bold")
         .tooltip("Delete original full images after processing is complete"))
        (ui.switch("Enable").props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["processing"]["delete"], "enabled"))


def create_webapp_settings() -> None:
    """Create UI elements and config binding for web app settings.

    Frame rate and resolution of streamed frames are derived automatically from
    camera.fps and camera.image.resolution (via RESOLUTION_PRESETS) and are
    therefore not configurable here.
    """
    with ui.grid(columns="auto 1fr").classes("w-full gap-x-5 items-center"):

        c = FIELD_CONSTRAINTS["webapp.stream.quality"]
        ui.label("JPEG Quality").classes("font-bold").tooltip("JPEG quality of streamed frames")
        (ui.number(label="JPEG", placeholder=app.state.config.webapp.stream.quality,
                   min=c["min"], max=c["max"], precision=0, step=1,
                   validation={f"Required value between {c['min']}-{c['max']}":
                               lambda v, c=c: validate_number(v, c["min"], c["max"])})
         .bind_value(app.state.config_updates["webapp"]["stream"], "quality",
                     forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("Use HTTPS").classes("font-bold")
         .tooltip("Use HTTPS protocol (required for browser Geolocation API to get GPS location)"))
        (ui.switch("Enable", on_change=lambda e: ui.notification(
            "Protocol changes require a full web app restart to take effect.", type="warning", timeout=3)
            if e.value != app.state.config.webapp.https.enabled else None)
         .props("color=green").classes("font-bold")
         .bind_value(app.state.config_updates["webapp"]["https"], "enabled"))


def remove_wifi_network(network_row: ui.row) -> None:
    """Remove a specific network row from UI and config."""
    if network_row in app.state.wifi_networks_ui:
        if len(app.state.wifi_networks_ui) > 1:
            idx = app.state.wifi_networks_ui.index(network_row)

            if idx < len(app.state.config_updates["network"]["wifi"]):
                app.state.config_updates["network"]["wifi"].pop(idx)

            network_row.delete()
            app.state.wifi_networks_ui.pop(idx)
        else:
            ui.notification("At least one Wi-Fi network must be configured!", type="warning", timeout=2)


def add_wifi_network(networks_column: ui.column, ssid: str = "", password: str = "") -> None:
    """Add a new Wi-Fi network input field.

    Args:
        networks_column: NiceGUI column element to append the new network row to.
        ssid:            Pre-filled SSID value (empty string for new networks).
        password:        Pre-filled password value (empty string for new networks).
    """
    with networks_column:
        new_network = {"ssid": ssid, "password": password}
        app.state.config_updates["network"]["wifi"].append(new_network)
        idx = len(app.state.config_updates["network"]["wifi"]) - 1

        with ui.row(align_items="baseline").classes("w-full gap-2") as network_row:
            (ui.input(label="SSID").props("clearable").classes("flex-1")
             .bind_value(app.state.config_updates["network"]["wifi"][idx], "ssid",
                         forward=lambda v: str(v) if v not in (None, "") else None))
            (ui.input(label="Password", validation={
                "Minimum 8 characters": lambda v: v is None or v == "" or len(str(v)) >= 8})
             .props("clearable").classes("flex-1")
             .bind_value(app.state.config_updates["network"]["wifi"][idx], "password",
                         forward=lambda v: str(v) if v not in (None, "") else None))
            ui.button(color="red", icon="delete",
                      on_click=lambda: remove_wifi_network(network_row)).props("round")

    app.state.wifi_networks_ui.append(network_row)


def create_network_settings() -> None:
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
                             forward=lambda v: str(v) if v not in (None, "") else None))
                (ui.input(label="Password", validation={
                    "Minimum 8 characters": lambda v: v is None or v == "" or len(str(v)) >= 8})
                 .props("clearable").classes("flex-1")
                 .bind_value(app.state.config_updates["network"]["hotspot"], "password",
                             forward=lambda v: str(v) if v not in (None, "") else None))

        grid_separator()
        (ui.label("Wi-Fi Networks").classes("font-bold")
         .tooltip("List of Wi-Fi networks that the RPi should connect to (ordered by priority)"))
        wifi_column = ui.column().classes("w-full gap-2")

        with wifi_column:
            app.state.config_updates["network"]["wifi"].clear()
            networks_column = ui.column().classes("w-full gap-2")
            for network in app.state.config.network.wifi:
                add_wifi_network(networks_column, network.ssid or "", network.password or "")
            ui.button("Add Wi-Fi", color="green", icon="add",
                      on_click=lambda: add_wifi_network(networks_column))


def create_system_settings() -> None:
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
                    c = FIELD_CONSTRAINTS["powermanager.charge_min"]
                    (ui.number(label="Min. Charge", placeholder=app.state.config.powermanager.charge_min,
                               min=c["min"], max=c["max"], precision=0, step=5, suffix="%",
                               validation={f"Required value between {c['min']}-{c['max']}":
                                           lambda v, c=c: validate_number(v, c["min"], c["max"])})
                     .classes("flex-1")
                     .tooltip("Minimum required charge level to start and continue a recording")
                     .bind_value(app.state.config_updates["powermanager"], "charge_min",
                                 forward=lambda v: int(v) if v is not None else None))
                    c = FIELD_CONSTRAINTS["powermanager.charge_check"]
                    (ui.number(label="Check Interval", placeholder=app.state.config.powermanager.charge_check,
                               min=c["min"], max=c["max"], precision=0, step=5, suffix="seconds",
                               validation={f"Required value between {c['min']}-{c['max']}":
                                           lambda v, c=c: validate_number(v, c["min"], c["max"])})
                     .classes("flex-1")
                     .bind_value(app.state.config_updates["powermanager"], "charge_check",
                                 forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("OAK Temperature").classes("font-bold")
         .tooltip("Maximum allowed OAK chip temperature to continue a recording"))
        with ui.row(align_items="center").classes("w-full gap-2"):
            c = FIELD_CONSTRAINTS["oak.temp_max"]
            (ui.number(label="Max. Temperature", placeholder=app.state.config.oak.temp_max,
                       min=c["min"], max=c["max"], precision=0, step=1, suffix="°C",
                       validation={f"Required value between {c['min']}-{c['max']}":
                                   lambda v, c=c: validate_number(v, c["min"], c["max"])})
             .classes("flex-1")
             .bind_value(app.state.config_updates["oak"], "temp_max",
                         forward=lambda v: int(v) if v is not None else None))
            c = FIELD_CONSTRAINTS["oak.temp_check"]
            (ui.number(label="Check Interval", placeholder=app.state.config.oak.temp_check,
                       min=c["min"], max=c["max"], precision=0, step=5, suffix="seconds",
                       validation={f"Required value between {c['min']}-{c['max']}":
                                   lambda v, c=c: validate_number(v, c["min"], c["max"])})
             .classes("flex-1")
             .bind_value(app.state.config_updates["oak"], "temp_check",
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
        (ui.label("System Metrics").classes("font-bold")
         .tooltip("Log system metrics (temperature, CPU/RAM utilization, battery info)"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["metrics"], "enabled"))
            c = FIELD_CONSTRAINTS["metrics.interval"]
            (ui.number(label="Log Interval", placeholder=app.state.config.metrics.interval,
                       min=c["min"], max=c["max"], precision=0, step=1, suffix="seconds",
                       validation={f"Required value between {c['min']}-{c['max']}":
                                   lambda v, c=c: validate_number(v, c["min"], c["max"])}).classes("w-full")
             .bind_visibility_from(app.state.config_updates["metrics"], "enabled")
             .bind_value(app.state.config_updates["metrics"], "interval",
                         forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        ui.label("Storage Management").classes("font-bold")
        with ui.row(align_items="center").classes("w-full gap-2"):
            c = FIELD_CONSTRAINTS["storage.disk_min"]
            (ui.number(label="Min. Free Space", placeholder=app.state.config.storage.disk_min,
                       min=c["min"], max=c["max"], precision=0, step=100, suffix="MB",
                       validation={f"Required value between {c['min']}-{c['max']} MB":
                                   lambda v, c=c: validate_number(v, c["min"], c["max"])}).classes("flex-1")
             .tooltip("Minimum required free disk space to start/continue a recording")
             .bind_value(app.state.config_updates["storage"], "disk_min",
                         forward=lambda v: int(v) if v is not None else None))
            c = FIELD_CONSTRAINTS["storage.disk_check"]
            (ui.number(label="Check Interval", placeholder=app.state.config.storage.disk_check,
                       min=c["min"], max=c["max"], precision=0, step=5, suffix="seconds",
                       validation={f"Required value between {c['min']}-{c['max']}":
                                   lambda v, c=c: validate_number(v, c["min"], c["max"])})
             .classes("flex-1")
             .bind_value(app.state.config_updates["storage"], "disk_check",
                         forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("Archive Data").classes("font-bold")
         .tooltip("Copy all captured data + logs/configs to archive directory and manage disk space"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["storage"]["archive"], "enabled"))
            c = FIELD_CONSTRAINTS["storage.archive.disk_low"]
            (ui.number(label="Low Free Space",
                       placeholder=app.state.config.storage.archive.disk_low,
                       min=c["min"], max=c["max"], precision=0, step=100, suffix="MB",
                       validation={f"Required value between {c['min']}-{c['max']} MB":
                                   lambda v, c=c: validate_number(v, c["min"], c["max"])}).classes("w-full")
             .tooltip("Delete oldest original data directories when free disk space drops below this threshold")
             .bind_visibility_from(app.state.config_updates["storage"]["archive"], "enabled")
             .bind_value(app.state.config_updates["storage"]["archive"], "disk_low",
                         forward=lambda v: int(v) if v is not None else None))

        grid_separator()
        (ui.label("Upload to Cloud").classes("font-bold")
         .tooltip("Upload archived data to cloud storage provider via rclone (always runs archive first)"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["storage"]["upload"], "enabled"))
            (ui.select(["all", "full", "crops", "timelapse", "metadata"], label="Content").classes("w-full")
             .tooltip("Select content for upload: 'all' excludes overlay frames, all options include metadata")
             .bind_visibility_from(app.state.config_updates["storage"]["upload"], "enabled")
             .bind_value(app.state.config_updates["storage"]["upload"], "content"))


def create_startup_settings() -> None:
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
         .tooltip("Automatically run image capture or web app at boot"))
        with ui.column().classes("w-full gap-1"):
            (ui.switch("Enable").props("color=green").classes("font-bold")
             .bind_value(app.state.config_updates["startup"]["auto_run"], "enabled"))

            with (ui.column().classes("w-full")
                  .bind_visibility_from(app.state.config_updates["startup"]["auto_run"], "enabled")):
                (ui.select(["capture", "webapp"], label="Primary Script").classes("w-full truncate")
                 .tooltip("Python script that is launched immediately on startup")
                 .bind_value(app.state.config_updates["startup"]["auto_run"], "primary"))
                (ui.select(["capture", "webapp", "None"], label="Fallback Script").classes("w-full truncate")
                 .tooltip("Python script launched after delay if primary is not interrupted by user interaction")
                 .bind_value(app.state.config_updates["startup"]["auto_run"], "fallback",
                             forward=lambda v: None if v == "None" else v,
                             backward=lambda v: "None" if v is None or v == "" else v))
                c = FIELD_CONSTRAINTS["startup.auto_run.delay"]
                (ui.number(label="Delay", placeholder=app.state.config.startup.auto_run.delay,
                           min=c["min"], max=c["max"], precision=0, step=1, suffix="seconds",
                           validation={f"Required value between {c['min']}-{c['max']}":
                                       lambda v, c=c: validate_number(v, c["min"], c["max"])}).classes("w-full")
                 .tooltip("If primary is 'webapp': cancel fallback if a live stream connection is detected "
                          "during this window. If primary is 'capture': always wait the full delay.")
                 .bind_value(app.state.config_updates["startup"]["auto_run"], "delay",
                             forward=lambda v: int(v) if v is not None else None))


async def update_sys_info() -> None:
    """Update system information from RPi and OAK camera."""
    rpi_info = get_rpi_metrics()
    oak_info = get_oak_metrics(app.state.q_syslog)
    if rpi_info:
        app.state.sys_info.update(rpi_info)
    if oak_info:
        app.state.sys_info.update(oak_info)


def create_sys_info_section() -> None:
    """Create UI elements for displaying RPi and OAK system information."""
    with ui.grid(columns="auto 1fr 1fr").classes("w-full gap-x-1 items-center"):
        ui.element()
        ui.label("RPi").classes("text-center text-h6 font-bold")
        ui.label("OAK").classes("text-center text-h6 font-bold")

        with ui.row(align_items="center").classes("gap-1"):
            ui.icon("thermostat", size="sm", color="orange")
            ui.label("TEMP")
        (ui.label().classes("text-center")
         .bind_text_from(app.state.sys_info, "rpi_cpu_temp", lambda t: f"CPU: {t} °C"))
        (ui.label().classes("text-center")
         .bind_text_from(app.state.sys_info, "oak_chip_temp", lambda t: f"Chip: {t} °C"))

        with ui.row(align_items="center").classes("gap-1"):
            ui.icon("speed", size="sm", color="blue")
            ui.label("CPU")
        with ui.column().classes("w-full gap-0"):
            (ui.label().classes("w-full text-center")
             .bind_text_from(app.state.sys_info, "rpi_cpu_usage_avg", lambda u: f"Average Usage: {u} %"))
            (ui.label().classes("w-full text-center")
             .bind_text_from(app.state.sys_info, "rpi_cpu_usage_sum", lambda u: f"Sum All Cores: {u} %"))
        with ui.column().classes("w-full gap-0"):
            (ui.label().classes("w-full text-center")
             .bind_text_from(app.state.sys_info, "oak_cpu_usage_css", lambda u: f"CSS Core: {u} %"))
            (ui.label().classes("w-full text-center")
             .bind_text_from(app.state.sys_info, "oak_cpu_usage_mss", lambda u: f"MSS Core: {u} %"))

        with ui.row(align_items="center").classes("gap-1"):
            ui.icon("memory", size="sm", color="green")
            ui.label("RAM")
        with ui.grid(columns="auto auto").classes("w-full gap-0 items-center"):
            ui.label("Usage").classes("text-center mb-1")
            ui.label("Free").classes("text-center mb-1")
            (ui.label().classes("text-center text-xs")
             .bind_text_from(app.state.sys_info, "rpi_ram_usage", lambda u: f"{u} %"))
            (ui.label().classes("text-center text-xs")
             .bind_text_from(app.state.sys_info, "rpi_ram_available", lambda a: f"{a} MB"))
        with ui.grid(columns="auto auto").classes("w-full gap-0 items-center"):
            ui.label("Usage").classes("text-center mb-1")
            ui.label("Free").classes("text-center mb-1")
            (ui.label().classes("text-center text-xs")
             .bind_text_from(app.state.sys_info, "oak_ram_usage_ddr", lambda u: f"DDR: {u} %"))
            (ui.label().classes("text-center text-xs")
             .bind_text_from(app.state.sys_info, "oak_ram_available_ddr", lambda a: f"{a} MB"))
            (ui.label().classes("text-center text-xs")
             .bind_text_from(app.state.sys_info, "oak_ram_usage_css", lambda u: f"CSS: {u} %"))
            (ui.label().classes("text-center text-xs")
             .bind_text_from(app.state.sys_info, "oak_ram_available_css", lambda a: f"{a} MB"))
            (ui.label().classes("text-center text-xs")
             .bind_text_from(app.state.sys_info, "oak_ram_usage_mss", lambda u: f"MSS: {u} %"))
            (ui.label().classes("text-center text-xs")
             .bind_text_from(app.state.sys_info, "oak_ram_available_mss", lambda a: f"{a} MB"))
            (ui.label().classes("text-center text-xs")
             .bind_text_from(app.state.sys_info, "oak_ram_usage_cmx", lambda u: f"CMX: {u} %"))
            (ui.label().classes("text-center text-xs")
             .bind_text_from(app.state.sys_info, "oak_ram_available_cmx", lambda a: f"{a} MB"))


def read_split_log(log_path: Path, max_lines: int = 1000) -> list[str]:
    """Read log file and return the last requested lines.

    Args:
        log_path:  Path to the log file.
        max_lines: Maximum number of lines to return (from the end of the file).

    Returns:
        List of the last max_lines lines from the log file.
    """
    with open(log_path, "r", encoding="utf-8") as log_file:
        return list(deque(log_file, maxlen=max_lines))


async def update_log_content(selected_log: str | None, log_display: ui.log) -> None:
    """Update content of log element based on selected log file."""
    if not selected_log:
        return

    log_display.clear()

    try:
        log_lines = await run.io_bound(read_split_log, LOGS_PATH / selected_log, max_lines=1000)
    except Exception as e:
        log_display.push(f"Error reading log file: {e}", classes="text-red")
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


def create_logs_section() -> None:
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


def create_terminal_section() -> None:
    """Create an interactive terminal element with a dedicated bash process via a pty.

    A new bash process is forked once per client connection via pty.fork().
    The pty and bash process are cleaned up when the client disconnects.

    WARNING: This gives any connected browser client full shell access to the RPi.
    """
    command_shortcuts: list[tuple[str, str, str, str, str]] = [
        # (label, icon, command, tooltip, color)
        ("WittyPi", "bolt",              "sudo ~/wittypi/wittyPi.sh\n", "Launch Witty Pi config", "blue"),
        ("htop",    "monitor_heart",     "htop\n",                      "Launch system monitor",  "teal"),
        ("df -h",   "storage",           "df -h\n",                     "Show disk space usage",  "cyan"),
        ("Clear",   "cleaning_services", "clear\n",                     "Clear terminal screen",  "grey"),
    ]

    # Mutable container so _send_command can reference pty_fd before it is assigned
    pty_fd_ref: list[int] = []

    def _send_command(command: str) -> None:
        """Write a pre-defined command string directly to the pty."""
        if not pty_fd_ref:
            return
        try:
            os.write(pty_fd_ref[0], command.encode("utf-8"))
        except OSError:
            pass

    with ui.column().classes("w-full gap-2"):
        with ui.row().classes("w-full flex-wrap gap-1"):
            for label, icon, command, tooltip, color in command_shortcuts:
                (ui.button(label, icon=icon, color=color,
                           on_click=lambda _, cmd=command: _send_command(cmd))
                 .props("dense outline size=sm")
                 .tooltip(tooltip))

        with ui.element("div").style("width: 100%; max-width: 100%; overflow-x: auto; contain: inline-size;"):
            terminal = ui.xterm(
                options={
                    "fontFamily": "'Cascadia Code', 'Courier New', 'Menlo', monospace",
                    "fontSize": 13,
                    "lineHeight": 1.2,
                    "cursorStyle": "block",
                    "cursorBlink": True,
                    "scrollback": 1000,
                    # Behaviour
                    "convertEol": True,
                    "scrollOnUserInput": True,
                    "macOptionIsMeta": True,
                    "altClickMovesCursor": True,
                    "cols": 80,
                    "rows": 24,
                    # Colors: VS Code Dark+ terminal color scheme
                    "theme": {
                        "background":          "#1e1e1e",
                        "foreground":          "#cccccc",
                        "cursor":              "#ffffff",
                        "cursorAccent":        "#000000",
                        "selectionBackground": "#264f78",
                        "black":               "#000000",
                        "red":                 "#cd3131",
                        "green":               "#0dbc79",
                        "yellow":              "#e5e510",
                        "blue":                "#2472c8",
                        "magenta":             "#bc3fbc",
                        "cyan":                "#11a8cd",
                        "white":               "#e5e5e5",
                        "brightBlack":         "#666666",
                        "brightRed":           "#f14c4c",
                        "brightGreen":         "#23d18b",
                        "brightYellow":        "#f5f543",
                        "brightBlue":          "#3b8eea",
                        "brightMagenta":       "#d670d6",
                        "brightCyan":          "#29b8db",
                        "brightWhite":         "#e5e5e5",
                    },
                }
            )

    try:
        pty_pid, pty_fd = pty.fork()  # type: ignore
    except Exception as e:
        ui.label(f"Failed to open terminal: {e}").classes("text-red")
        logger.error("Failed to fork pty for terminal: %s", e)
        return

    if pty_pid == pty.CHILD:  # type: ignore
        # Child process: set TERM/COLORTERM for full color support and launch login bash
        # so PATH and other env vars are set correctly, matching SSH login behaviour
        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        os.execve("/bin/bash", ["/bin/bash", "--login"], env)

    # Populate the mutable container so _send_command can now write to the pty
    pty_fd_ref.append(pty_fd)
    logger.debug("Terminal pty opened (pid=%d, fd=%d)", pty_pid, pty_fd)

    def pty_to_terminal() -> None:
        """Read data from the pty and write it to the browser terminal."""
        try:
            data = os.read(pty_fd, 4096)
        except OSError:
            logger.info("Terminal pty fd closed (pid=%d)", pty_pid)
            if core.loop is not None:
                core.loop.remove_reader(pty_fd)
        else:
            terminal.write(data)

    if core.loop is not None:
        core.loop.add_reader(pty_fd, pty_to_terminal)

    @terminal.on_data
    def terminal_to_pty(event: XtermDataEventArguments) -> None:
        """Write user input from the browser terminal to the pty."""
        try:
            os.write(pty_fd, event.data.encode("utf-8"))
        except OSError:
            pass

    @ui.context.client.on_delete
    def kill_bash() -> None:
        """Clean up pty and bash process when the browser client disconnects."""
        pty_fd_ref.clear()
        try:
            if core.loop is not None:
                core.loop.remove_reader(pty_fd)
        except Exception:
            pass
        try:
            os.close(pty_fd)
        except OSError:
            pass
        try:
            os.kill(pty_pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except ProcessLookupError:
            pass
        logger.debug("Terminal pty closed (pid=%d)", pty_pid)


async def apply_config_changes(
    config_name: str,
    has_network_changes: bool,
    config_selected: AppConfig | None = None
) -> None:
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
                set_up_network(config_selected, activate_network=True)
            else:
                set_up_network(app.state.config, activate_network=True)
        except Exception as e:
            ui.notification(f"Network settings failed to apply: {e}", type="negative", timeout=5)
            return

    ui.notification(f"Activating configuration '{config_name}'...",
                    position="top", type="info", spinner=True, timeout=2)
    await asyncio.sleep(0.5)
    update_config_selector(config_name)
    app.state.config_active = config_name
    await close_camera()
    await asyncio.sleep(0.5)
    ui.navigate.reload()


async def show_apply_dialog(config_name: str, has_network_changes: bool) -> None:
    """Show dialog to apply changes to current config.

    Args:
        config_name:         Name of the config file that was saved.
        has_network_changes: True if network settings were changed and need to be applied.
    """
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


async def show_activate_dialog(config_name: str, has_network_changes: bool) -> None:
    """Show dialog to activate another config.

    Args:
        config_name:         Name of the config file that was saved.
        has_network_changes: True if network settings were changed and need to be applied.
    """
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


async def save_to_file(config_path: Path) -> None:
    """Save configuration to specified file path.

    Args:
        config_path: Path to the target config YAML file.
    """
    has_network_changes = check_config_changes(
        app.state.config.network.model_dump(),
        app.state.config_updates["network"]
    )

    app.state.config = update_config_yaml(config_path, app.state.config_updates)

    ui.notification(f"Configuration saved to '{config_path.name}'!", type="positive", timeout=2)

    app.state.configs = sorted([f.name for f in CONFIGS_PATH.glob("*.yaml")
                                if f.name != "config_selector.yaml"])

    if config_path.name == app.state.config_active:
        await show_apply_dialog(config_path.name, has_network_changes)
    else:
        app.state.config_updates = copy.deepcopy(app.state.config.model_dump())
        await show_activate_dialog(config_path.name, has_network_changes)


async def config_name_input() -> str:
    """Show dialog to enter a name for the new configuration file.

    Returns:
        Entered filename (without .yaml extension), 'cancel' if cancelled,
        or empty string if no valid name was entered.
    """
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


async def create_new_config() -> None:
    """Create a new configuration file."""
    filename = await config_name_input()
    if not filename:
        ui.notification("Please enter a valid filename!", type="warning", timeout=2)
        return
    if filename == "cancel":
        ui.notification("New config creation cancelled!", type="warning", timeout=2)
        return

    config_new_path = CONFIGS_PATH / f"{filename}.yaml"
    if config_new_path.exists():
        if filename == "config_selector":
            ui.notification("Cannot overwrite config_selector.yaml!", type="warning", timeout=2)
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


async def save_config() -> None:
    """Save modified configuration by overwriting the current config file or creating a new one."""
    config_current_path = CONFIGS_PATH / app.state.config_active

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


async def start_recording() -> None:
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
        if action == "save":
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


async def confirm_shutdown() -> None:
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
        if action == "save":
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


async def disconnect() -> None:
    """Disconnect all clients from currently running server."""
    for client_id in Client.instances:
        await core.sio.disconnect(client_id)


async def on_app_shutdown() -> None:
    """Disconnect clients, close running OAK device, and optionally start recording session."""
    await disconnect()
    await close_camera()

    # Remove streaming marker after web app exits
    STREAMING_MARKER.unlink(missing_ok=True)

    if getattr(app.state, "start_recording", False):
        subprocess_log("capture.py")

        with open(LOGS_PATH / "subprocess.log", "a", encoding="utf-8") as log_file_handle:
            subprocess.Popen(
                [str(UV), "run", "capture"],
                stdout=log_file_handle,
                stderr=log_file_handle,
                cwd=BASE_PATH,
                start_new_session=True
            )

    try:
        loop = asyncio.get_running_loop()
        def _force_exit() -> None:
            """Force exit the web app if it hasn't exited within the timeout."""
            logger.warning("Web app forced to exit after timeout.")
            sys.exit(0)
        loop.call_later(10, _force_exit)
    except RuntimeError:
        logger.warning("No running event loop found, exiting immediately.")
        sys.exit(0)


def signal_handler(
    loop: asyncio.AbstractEventLoop,
    signum: int,
    frame: FrameType | None
) -> None:
    """Handle a received signal to gracefully shut down the app.

    Args:
        loop:   Running asyncio event loop.
        signum: Signal number received.
        frame:  Current stack frame at the time of the signal (may be None).
    """
    logger.info("Signal received, initiating graceful app shutdown...")
    loop.create_task(close_camera())
    app.shutdown()


def on_app_startup(
    protocol: str,
    port: int,
    use_https: bool
) -> None:
    """Register signal handler and show startup message.

    Args:
        protocol:  Protocol string ('http' or 'https').
        port:      Port number the web app is listening on.
        use_https: True if HTTPS is enabled and SSL certificates are present.
    """
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: signal_handler(loop, signal.SIGINT, None))
    loop.add_signal_handler(signal.SIGTERM, lambda: signal_handler(loop, signal.SIGTERM, None))

    logger.info("Insect Detect web app ready to go!")
    logger.info("Access via hostname:   %s://%s:%s", protocol, HOSTNAME, port)
    logger.info("Access via IP address: %s://%s:%s", protocol, IP_ADDRESS, port)
    if use_https:
        logger.info("Accept the self-signed SSL certificate in your browser when first connecting.")


def main() -> None:
    """Main function to start the web app."""
    # Create directory for storing logs
    LOGS_PATH.mkdir(parents=True, exist_ok=True)

    # Configure logging (write logs to file and stream to console)
    configure_logger(Path(__file__).stem, stream_to_console=True)
    logger.info("-------- Web App Logger initialized --------")

    # Parse config and set parameters based on HTTPS setting
    config_selector = load_config_selector()
    config_active = config_selector.config_active
    config = load_config_yaml(CONFIGS_PATH / config_active)
    https_enabled = config.webapp.https.enabled
    ssl_cert_path = BASE_PATH / "ssl" / "cert.pem"
    ssl_key_path = BASE_PATH / "ssl" / "key.pem"
    use_https = https_enabled and ssl_cert_path.exists() and ssl_key_path.exists()
    if https_enabled and not use_https:
        logger.warning("HTTPS is enabled but no SSL certificates were found. Using HTTP instead.")
    protocol = "https" if use_https else "http"
    port = 8443 if use_https else 5000
    ssl_cert = str(ssl_cert_path) if use_https else None
    ssl_key = str(ssl_key_path) if use_https else None

    # Increase threshold for max. binding propagation time to avoid early warning messages
    binding.MAX_PROPAGATION_TIME = 0.05  # default: 0.01 seconds

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


if __name__ == "__main__":
    main()
