"""Capture detection-triggered images and save model/tracker metadata from OAK camera.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Usage:
    Run with 'uv run capture' from the insect-detect directory ('cd insect-detect').
    Configure settings via 'configs/config.yaml' (select active config in 'config_selector.yaml').

Recording Session Flow:
    1. Initialize power manager (Witty Pi 4 L3V7 or PiJuice Zero) if enabled.
    2. Check preconditions: disk space and battery charge must meet thresholds (else: shutdown).
    3. Create timestamped session directory and save config snapshot.
    4. Run OAK object detection + tracking pipeline on downscaled frames.
    5. Synchronize tracker output with high-resolution frames.
    6. Capture high-resolution frames on detection triggers and/or timelapse intervals.
    7. Save MJPEG-encoded frames and corresponding metadata (.csv) to session directory.
    8. Process captured images in real-time (crop bboxes, draw overlays) if enabled.
    9. Archive data and optionally upload to cloud storage if enabled.

Stop Conditions:
    - Configured recording session duration exceeded.
    - Free disk space drops below threshold.
    - OAK chip temperature exceeds threshold.
    - Battery charge level drops below threshold more than twice.
    - External shutdown trigger (e.g. button press).
    - OAK pipeline stops unexpectedly.

Recording Duration:
    Automatically adjusted based on current battery charge level (high/medium/low).
    Falls back to default duration if power management is disabled.
"""

import csv
import json
import logging
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import depthai as dai
import psutil
from gpiozero import LED

from insectdetect.config import AppConfig, load_config_selector, load_config_yaml, sanitize_config
from insectdetect.constants import CONFIGS_PATH, DATA_PATH, HOSTNAME, LOGS_PATH
from insectdetect.data import archive_data, save_encoded_frame, upload_data
from insectdetect.metrics import configure_logger, save_metrics, save_session_info
from insectdetect.oak import create_pipeline, deletterbox_bbox
from insectdetect.postprocess import process_images
from insectdetect.power import PowerManagerState, init_power_manager

# Initialize logger for this module
logger = logging.getLogger(__name__)


@dataclass
class RecordingContext:
    """Context object holding all state and resources for a single recording session."""
    session_path: Path
    metadata_path: Path
    timelapse_path: Path
    session_id: int
    session_dur: float
    pwr: PowerManagerState
    pipeline: dai.Pipeline
    q_frames: dai.MessageQueue
    q_tracks: dai.MessageQueue
    q_syslog: dai.MessageQueue
    q_camctrl: dai.InputQueue
    frame_size: tuple[int, int]
    nn_input_size: tuple[int, int]
    sensor_roi: tuple[int, int, int, int]
    labels: list[str]


@dataclass
class RecordingResult:
    """Final system state and stop reason flags at the end of a recording session."""
    session_start: datetime | None = None
    session_end: datetime | None = None
    disk_free: int = 0
    chargelevel: int | str | None = None
    temp_oak: int = 0
    stopped_by_shutdown: bool = False
    stopped_by_disk: bool = False
    stopped_by_temp: bool = False
    stopped_by_charge: bool = False


def _start_led(config: AppConfig) -> LED | None:
    """Initialize and start LED to indicate recording session is running.

    Retries initialization for up to 2 seconds to allow for GPIO availability
    delays at startup. Returns None if LED is disabled or initialization fails.

    Args:
        config: AppConfig containing all configuration settings.

    Returns:
        Active LED instance if successfully initialized, None otherwise.
    """
    if not config.led.enabled:
        return None
    for _ in range(20):
        try:
            led = LED(config.led.gpio_pin)
            led.on()
            return led
        except Exception:
            time.sleep(0.1)
    logger.warning("Could not initialize LED on GPIO pin %s", config.led.gpio_pin)
    return None


def _check_preconditions(config: AppConfig, pwr: PowerManagerState, disk_free: int) -> None:
    """Shut down Raspberry Pi if disk space or battery charge is too low to start recording session.

    Args:
        config:    AppConfig containing all configuration settings.
        pwr:       PowerManagerState with charge level and threshold.
        disk_free: Current free disk space in MB.
    """
    if disk_free < config.storage.disk_min:
        logger.warning("Shut down without recording due to low disk space: %s MB", disk_free)
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
        sys.exit(0)
    elif pwr.enabled:
        charge_too_low = (
            pwr.chargelevel_start == "NA"
            or (isinstance(pwr.chargelevel_start, int)
                and pwr.chargelevel_start < config.powermanager.charge_min)
        )
        if charge_too_low:
            logger.warning("Shut down without recording due to low charge level: %s%%",
                           pwr.chargelevel_start)
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
            sys.exit(0)


def _get_session_dur(config: AppConfig, pwr: PowerManagerState) -> float:
    """Determine the recording session duration based on the current battery charge level.

    If power management is disabled or battery charge cannot be read,
    the configured default recording session duration is used.

    Charge level thresholds:
    - >= 70% or USB_C_IN: high duration
    - 50-69%:             medium duration
    - 30-49%:             low duration

    Args:
        config: AppConfig containing all configuration settings.
        pwr:    PowerManagerState with current charge level.

    Returns:
        Recording session duration in seconds.
    """
    if pwr.enabled:
        if pwr.chargelevel_start == "USB_C_IN" or (
            isinstance(pwr.chargelevel_start, int) and pwr.chargelevel_start >= 70
        ):
            return config.recording.duration.battery.high * 60
        if isinstance(pwr.chargelevel_start, int) and pwr.chargelevel_start >= 50:
            return config.recording.duration.battery.medium * 60
        if isinstance(pwr.chargelevel_start, int) and pwr.chargelevel_start >= 30:
            return config.recording.duration.battery.low * 60
    return config.recording.duration.default * 60


def _create_session_dir(
    data_path: Path,
    config_active: str,
    config: AppConfig
) -> tuple[Path, Path, Path, int]:
    """Create timestamped directory for this recording session and save config snapshot.

    The session ID is read from a persistent file and incremented on each run.
    The active config is saved as a sanitized JSON snapshot (passwords masked)
    alongside the captured data for reproducibility.

    Args:
        data_path:     Root data directory where session folders are created.
        config_active: Filename of the active config (e.g. 'config.yaml').
        config:        AppConfig containing all configuration settings.

    Returns:
        Tuple of (session_path, metadata_path, timelapse_path, session_id).
    """
    session_id_file = data_path / "last_session_id.txt"
    session_id = int(session_id_file.read_text(encoding="utf-8")) + 1 if session_id_file.exists() else 1
    session_id_file.write_text(str(session_id), encoding="utf-8")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_path = data_path / timestamp[:10] / timestamp
    session_path.mkdir(parents=True, exist_ok=True)
    metadata_path = session_path / f"{timestamp}_metadata.csv"
    timelapse_path = session_path / "timelapse"
    timelapse_path.mkdir(exist_ok=True)

    config_snapshot_path = session_path / f"{timestamp}_{Path(config_active).stem}.json"
    json.dump(sanitize_config(config), config_snapshot_path.open("w", encoding="utf-8"), indent=2)

    logger.info("Recording session directory created: %s", session_path)
    return session_path, metadata_path, timelapse_path, session_id


def _start_metrics_thread(
    stop_event: threading.Event,
    interval: float,
    initial_delay: float,
    session_path: Path,
    device_id: str,
    session_id: int,
    q_syslog: dai.MessageQueue,
    get_power_info: Callable[[], dict[str, object]] | None
) -> threading.Thread:
    """Start a background thread that saves system metrics at a fixed interval.

    The thread waits for the initial delay before the first call, then calls
    save_metrics() repeatedly at the configured interval until stop_event is set.
    Uses stop_event.wait() instead of time.sleep() so the thread wakes up
    immediately when stop_event is set, avoiding shutdown delays.

    Args:
        stop_event:     Event that signals the thread to stop.
        interval:       Interval in seconds between save_metrics() calls.
        initial_delay:  Delay in seconds before the first save_metrics() call.
        session_path:   Recording session directory where the log file is written.
        device_id:      Camera trap ID (hostname).
        session_id:     Incrementing recording session counter.
        q_syslog:       depthai output queue for SystemInformation messages.
        get_power_info: Optional callable returning a dict of power metrics.

    Returns:
        Started daemon thread running the metrics saving loop.
    """
    def _metrics_loop() -> None:
        stop_event.wait(timeout=initial_delay)
        while not stop_event.is_set():
            save_metrics(session_path, device_id, session_id, q_syslog, get_power_info)
            stop_event.wait(timeout=interval)

    thread = threading.Thread(target=_metrics_loop, name="MetricsLogger", daemon=True)
    thread.start()
    return thread


def _run_recording(
    ctx: RecordingContext,
    config: AppConfig,
    disk_free: int
) -> RecordingResult:
    """Run the main recording loop and return a RecordingResult with stop reason flags.

    Starts the OAK pipeline, optionally starts metrics logging and image processing
    threads, then runs the frame capture loop until a stop condition is met.
    Cleans up all resources and writes the session summary on exit.

    Args:
        ctx:       RecordingContext with all session state and resources.
        config:    AppConfig containing all configuration settings.
        disk_free: Initial free disk space in MB (measured before session start).

    Returns:
        RecordingResult with stop reason flags and final system state.
    """
    result = RecordingResult()

    with (
        ctx.pipeline,
        ThreadPoolExecutor(max_workers=1) as executor,
        open(ctx.metadata_path, "a", buffering=1, encoding="utf-8") as metadata_file
    ):
        # Start optional system metrics logging thread
        metrics_thread: threading.Thread | None = None
        metrics_stop = threading.Event()
        if config.metrics.enabled:
            metrics_thread = _start_metrics_thread(
                stop_event=metrics_stop,
                interval=config.metrics.interval,
                initial_delay=4.0,
                session_path=ctx.session_path,
                device_id=HOSTNAME,
                session_id=ctx.session_id,
                q_syslog=ctx.q_syslog,
                get_power_info=ctx.pwr.get_power_info
            )

        # Start optional image post-processing thread
        processing_thread: threading.Thread | None = None
        processing_stop = threading.Event()
        metadata_queue: queue.Queue[list[dict[str, object]]] | None = None
        if config.processing.crop.enabled or config.processing.overlay.enabled:
            metadata_queue = queue.Queue()
            processing_thread = threading.Thread(
                target=process_images,
                args=(metadata_queue, ctx.session_path, config, processing_stop),
                name="ImageProcessor",
                daemon=False
            )
            processing_thread.start()

        # Write header to metadata .csv file
        metadata_writer = csv.DictWriter(metadata_file, fieldnames=[
            "device_id", "session_id", "timestamp", "label", "confidence",
            "track_id", "track_status", "x_min", "y_min", "x_max", "y_max",
            "lens_position", "iso_sensitivity", "exposure_time", "filename"
        ])
        metadata_writer.writeheader()

        # Upload pipeline to OAK device and start camera
        ctx.pipeline.start()

        # Ensure camera runs in continuous auto focus mode if manual focus is not enabled
        # Workaround to avoid focus issues if setAutoFocusRegion() was used in initial camera control
        if config.camera.focus.mode != "manual":
            af_ctrl = dai.CameraControl().setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            ctx.q_camctrl.send(af_ctrl)

        # Give auto focus/exposure/whitebalance some time to stabilize
        time.sleep(2)

        # Log recording session start with initial system state
        if ctx.pwr.enabled:
            logger.info(
                "Recording session %s started | Duration: %s min | Free disk space: %s MB | Charge level: %s%%",
                ctx.session_id, round(ctx.session_dur / 60), disk_free, ctx.pwr.chargelevel_start
            )
        else:
            logger.info(
                "Recording session %s started | Duration: %s min | Free disk space: %s MB",
                ctx.session_id, round(ctx.session_dur / 60), disk_free
            )

        # Initialize variables for capture/check events at start of recording session
        det_interval = config.recording.interval.detection
        tl_interval = config.recording.interval.timelapse
        ae_region = config.detection.ae_region.enabled
        ae_region_active = False
        last_ae_time: float = 0.0
        disk_min = config.storage.disk_min
        disk_check = config.storage.disk_check
        temp_oak_max = config.oak.temp_max
        temp_oak_check = config.oak.temp_check
        charge_min = config.powermanager.charge_min
        charge_check = config.powermanager.charge_check
        sysinfo = cast(dai.SystemInformation | None, ctx.q_syslog.tryGet())
        result.disk_free = disk_free
        result.temp_oak = round(sysinfo.chipTemperature.average) if sysinfo else 0
        result.chargelevel = ctx.pwr.chargelevel_start if ctx.pwr.enabled else None
        result.session_start = datetime.now()
        start_time = time.monotonic()
        last_capture = start_time - tl_interval  # trigger first timelapse capture immediately
        next_capture = start_time + det_interval
        last_disk_check = start_time
        last_temp_oak_check = start_time
        last_charge_check = start_time
        chargelevels: list[int | str | None] = []

        try:
            # Run recording session as long as all conditions are met
            while (
                time.monotonic() < start_time + ctx.session_dur
                and ctx.pipeline.isRunning()
                and not ctx.pwr.external_shutdown.is_set()
                and result.disk_free > disk_min
                and result.temp_oak < temp_oak_max
                and len(chargelevels) < 3
            ):
                # Determine whether to capture image based on current time and configured intervals
                track_active = False
                current_time = time.monotonic()
                triggered_capture = current_time >= next_capture
                timelapse_capture = current_time >= last_capture + tl_interval
                frame = cast(dai.ImgFrame | None, ctx.q_frames.tryGet())
                track = cast(dai.Tracklets | None, ctx.q_tracks.tryGet())

                if frame is not None and (triggered_capture or timelapse_capture):
                    timestamp = datetime.now()
                    timestamp_iso = timestamp.isoformat()
                    timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")
                    file_stem = f"{HOSTNAME}_{timestamp_str}"
                    lens_pos: int = frame.getLensPosition()
                    iso_sens: int = frame.getSensitivity()
                    exp_time: float = frame.getExposureTime().total_seconds() * 1000
                    frame_metadata: list[dict[str, object]] = []

                    if track is not None:
                        tracklet_id_max = -1
                        bbox_max: tuple[float, float, float, float] | None = None

                        for tracklet in track.tracklets:
                            # Only process active tracklets (not "LOST" or "REMOVED")
                            tracklet_status = tracklet.status.name
                            if tracklet_status in {"TRACKED", "NEW"}:
                                track_active = True
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
                                    ctx.frame_size[0], ctx.frame_size[1],
                                    ctx.nn_input_size[0], ctx.nn_input_size[1]
                                )

                                # Save metadata from camera and tracker + model output to .csv
                                metadata: dict[str, object] = {
                                    "device_id": HOSTNAME,
                                    "session_id": ctx.session_id,
                                    "timestamp": timestamp_iso,
                                    "label": ctx.labels[tracklet.srcImgDetection.label],
                                    "confidence": round(tracklet.srcImgDetection.confidence, 2),
                                    "track_id": tracklet_id,
                                    "track_status": tracklet_status,
                                    "x_min": round(bbox[0], 4),
                                    "y_min": round(bbox[1], 4),
                                    "x_max": round(bbox[2], 4),
                                    "y_max": round(bbox[3], 4),
                                    "lens_position": lens_pos,
                                    "iso_sensitivity": iso_sens,
                                    "exposure_time": round(exp_time, 2),
                                    "filename": f"{file_stem}.jpg"
                                }
                                metadata_writer.writerow(metadata)
                                frame_metadata.append(metadata)

                                # Track most recent active tracking ID and its bounding box
                                if tracklet_status == "TRACKED" and tracklet_id > tracklet_id_max:
                                    tracklet_id_max = tracklet_id
                                    bbox_max = bbox

                        if ae_region:
                            if bbox_max:
                                ae_time = time.monotonic()
                                if ae_time - last_ae_time >= 1.0:
                                    # Set AE region to bbox of most recent active tracking ID (capped to 1 Hz)
                                    # Map bbox (frame-normalized) to sensor-space coordinates
                                    roi_x, roi_y, roi_w, roi_h = ctx.sensor_roi
                                    rect_bbox: tuple[int, int, int, int] = (
                                        max(1, round(roi_x + bbox_max[0] * roi_w)),
                                        max(1, round(roi_y + bbox_max[1] * roi_h)),
                                        max(10, round((bbox_max[2] - bbox_max[0]) * roi_w)),
                                        max(10, round((bbox_max[3] - bbox_max[1]) * roi_h))
                                    )
                                    exp_ctrl = dai.CameraControl().setAutoExposureRegion(*rect_bbox)
                                    ctx.q_camctrl.send(exp_ctrl)
                                    ae_region_active = True
                                    last_ae_time = ae_time
                            elif ae_region_active:
                                # Reset AE region to full visible FOV in sensor space if no active tracking ID
                                exp_ctrl = dai.CameraControl().setAutoExposureRegion(*ctx.sensor_roi)
                                ctx.q_camctrl.send(exp_ctrl)
                                ae_region_active = False
                                last_ae_time = 0.0

                    if track_active or timelapse_capture:
                        # Save MJPEG-encoded frame to .jpg in a separate thread
                        trigger = "detection" if track_active else "timelapse"
                        executor.submit(save_encoded_frame, frame, ctx.session_path, file_stem, trigger)
                        last_capture = current_time
                        next_capture = current_time + det_interval

                        # Update free disk space (MB) at configured interval
                        if current_time >= last_disk_check + disk_check:
                            result.disk_free = round(psutil.disk_usage("/").free / 1048576)
                            last_disk_check = current_time

                    # Put latest frame metadata in the queue for the image processing thread
                    if metadata_queue is not None and frame_metadata:
                        metadata_queue.put_nowait(frame_metadata)

                # Update OAK chip temperature at configured interval
                if current_time >= last_temp_oak_check + temp_oak_check:
                    sysinfo = cast(dai.SystemInformation | None, ctx.q_syslog.tryGet())
                    result.temp_oak = round(sysinfo.chipTemperature.average) if sysinfo else result.temp_oak
                    last_temp_oak_check = current_time

                # Update charge level at configured interval and add to list if below threshold
                if ctx.pwr.enabled and current_time >= last_charge_check + charge_check:
                    result.chargelevel = ctx.pwr.get_chargelevel()
                    charge_too_low = (
                        result.chargelevel == "NA"
                        or (isinstance(result.chargelevel, int)
                            and result.chargelevel < charge_min)
                    )
                    if charge_too_low:
                        chargelevels.append(result.chargelevel)
                    last_charge_check = current_time

                # Sleep briefly to avoid busy-waiting (max. 20 FPS)
                time.sleep(0.05)

            # Determine stop reason and log recording session end
            result.stopped_by_shutdown = ctx.pwr.external_shutdown.is_set()
            result.stopped_by_disk = result.disk_free < disk_min
            result.stopped_by_temp = result.temp_oak >= temp_oak_max
            result.stopped_by_charge = ctx.pwr.enabled and len(chargelevels) >= 3
            _log_recording_end(ctx, result)

        except Exception:
            logger.exception("Error during recording session %s", ctx.session_id)
        finally:
            # Stop background metrics logger and wait for thread to finish
            metrics_stop.set()
            if metrics_thread is not None:
                metrics_thread.join(timeout=10)
                if metrics_thread.is_alive():
                    logger.warning("Metrics logger thread did not finish within 10 seconds")

            # Write recording session summary to .csv file
            result.session_end = datetime.now()
            save_session_info(
                DATA_PATH, ctx.session_path, HOSTNAME, ctx.session_id,
                result.session_start, result.session_end,
                ctx.pwr.chargelevel_start, result.chargelevel
            )

            # Remove timelapse directory if no frames were saved
            if ctx.timelapse_path.exists() and not next(ctx.timelapse_path.iterdir(), None):
                ctx.timelapse_path.rmdir()

            # Signal stop to image processing thread and wait for it to finish
            if processing_thread is not None:
                processing_stop.set()
                processing_thread.join(timeout=120)
                if processing_thread.is_alive():
                    logger.warning("Image processing thread did not finish within 2 minutes")

    return result


def _log_recording_end(ctx: RecordingContext, result: RecordingResult) -> None:
    """Log a summary message at the end of the recording session.

    Logs a warning for each early-stop condition, or an info message with
    duration and final system state if the session completed normally.

    Args:
        ctx:    RecordingContext with session ID, duration and power management state.
        result: RecordingResult with stop reason flags and final system state.
    """
    if result.stopped_by_shutdown:
        logger.warning("Recording session %s stopped early by external trigger", ctx.session_id)
    elif result.stopped_by_disk:
        logger.warning("Recording session %s stopped early due to low disk space: %s MB",
                       ctx.session_id, result.disk_free)
    elif result.stopped_by_temp:
        logger.warning("Recording session %s stopped early due to high OAK chip temperature: %s °C",
                       ctx.session_id, result.temp_oak)
    elif result.stopped_by_charge:
        logger.warning("Recording session %s stopped early due to low charge level: %s%%",
                       ctx.session_id, result.chargelevel)
    elif ctx.pwr.enabled:
        logger.info(
            "Recording session %s finished | Duration: %s min | Free disk space: %s MB | Charge level: %s%%",
            ctx.session_id, round(ctx.session_dur / 60), result.disk_free, result.chargelevel
        )
    else:
        logger.info(
            "Recording session %s finished | Duration: %s min | Free disk space: %s MB",
            ctx.session_id, round(ctx.session_dur / 60), result.disk_free
        )


def _archive_and_upload(
    ctx: RecordingContext,
    result: RecordingResult,
    config: AppConfig
) -> None:
    """Copy captured data to archive directory and optionally upload after recording session.

    Archiving is always performed when upload is enabled since uploading requires
    an archived copy of the data. Skipped if recording session was stopped early
    by external trigger, low disk space, or low battery charge.
    An additional charge level check is performed before archiving to ensure
    sufficient battery remains (at least charge_min + 10%) when not on USB power.

    Args:
        ctx:    RecordingContext with data path, device ID and power management state.
        result: RecordingResult with stop reason flags and final system state.
        config: AppConfig containing all configuration settings.
    """
    if not (config.storage.archive.enabled or config.storage.upload.enabled):
        return

    if result.stopped_by_shutdown:
        logger.warning("Skipped archiving/uploading: recording session was stopped by external trigger")
        return
    if result.stopped_by_disk:
        logger.warning("Skipped archiving/uploading: recording session was stopped due to low disk space: %s MB",
                       result.disk_free)
        return
    if result.stopped_by_charge:
        logger.warning("Skipped archiving/uploading: recording session was stopped due to low charge level: %s%%",
                       result.chargelevel)
        return

    power_ok = not ctx.pwr.enabled or (
        result.chargelevel == "USB_C_IN"
        or (isinstance(result.chargelevel, int)
            and result.chargelevel > config.powermanager.charge_min + 10)
    )
    if not power_ok:
        logger.warning("Skipped archiving/uploading: charge level too low for safe archiving: %s%%",
                       result.chargelevel)
        return

    try:
        archive_path = archive_data(DATA_PATH, HOSTNAME, config.storage.archive.disk_low)
        logger.info("Archiving of data finished")
        if config.storage.upload.enabled:
            upload_data(DATA_PATH, archive_path, config.storage.upload.content)
            logger.info("Uploading of data finished")
    except Exception:
        logger.exception("Error during archiving/uploading of data")


def main() -> None:
    """Run a recording session."""
    # Create data and logs directories if they don't exist
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)

    # Configure logging (write logs to file)
    configure_logger(Path(__file__).stem)
    logger.info("-------- Capture Logger initialized --------")

    # Load active configuration file
    config_selector = load_config_selector()
    config_active = config_selector.config_active
    config = load_config_yaml(CONFIGS_PATH / config_active)
    logger.info("Configuration '%s' loaded successfully", config_active)

    # Initialize power manager (Witty Pi 4 L3V7 or PiJuice Zero)
    pwr = PowerManagerState.disabled()
    if config.powermanager.enabled:
        try:
            pwr = init_power_manager(config.powermanager.model)
        except Exception:
            logger.exception("Error during initialization of power manager: power management disabled")

    # Shut down early if disk space or battery charge is too low to start recording session
    disk_free = round(psutil.disk_usage("/").free / 1048576)
    _check_preconditions(config, pwr, disk_free)

    # Turn on LED to indicate recording session is running
    led = _start_led(config)

    # Determine recording session duration based on battery charge level (or use default)
    session_dur = _get_session_dur(config, pwr)

    # Create timestamped session directory and save sanitized config snapshot
    session_path, metadata_path, timelapse_path, session_id = _create_session_dir(
        DATA_PATH, config_active, config
    )

    # Create OAK camera pipeline and output queues
    (pipeline, q_frames, q_tracks, q_syslog, q_camctrl,
     frame_size, nn_input_size, sensor_roi, labels) = create_pipeline(config, stream=False)

    # Bundle all session state into a single context object
    ctx = RecordingContext(
        session_path=session_path,
        metadata_path=metadata_path,
        timelapse_path=timelapse_path,
        session_id=session_id,
        session_dur=session_dur,
        pwr=pwr,
        pipeline=pipeline,
        q_frames=q_frames,
        q_tracks=q_tracks,
        q_syslog=q_syslog,
        q_camctrl=q_camctrl,
        frame_size=frame_size,
        nn_input_size=nn_input_size,
        sensor_roi=sensor_roi,
        labels=labels
    )

    try:
        # Start recording session
        result = _run_recording(ctx, config, disk_free)
        _archive_and_upload(ctx, result, config)
    except KeyboardInterrupt:
        logger.warning("Recording session %s stopped by Ctrl+C", session_id)
    except Exception:
        logger.exception("Error during initialization of recording session %s", session_id)
    finally:
        if led:
            led.off()
        # Optionally shut down Raspberry Pi after recording session
        if not pwr.external_shutdown.is_set() and config.recording.shutdown.enabled:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    main()
