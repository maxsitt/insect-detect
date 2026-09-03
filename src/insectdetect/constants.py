"""Shared constants used across insect-detect modules.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/
"""

import socket
from pathlib import Path
from typing import Literal, get_args

# Hostname used as device identifier (camera trap ID)
HOSTNAME: str = socket.gethostname()

# Path to uv executable
UV: Path = Path.home() / ".local" / "bin" / "uv"

# Paths to root directory and subdirectories for data, logs, configs, and models
BASE_PATH: Path = Path.home() / "insect-detect"
DATA_PATH: Path = BASE_PATH / "data"
ARCHIVE_PATH: Path = BASE_PATH / "data_archived" / HOSTNAME
LOGS_PATH: Path = BASE_PATH / "logs"
CONFIGS_PATH: Path = BASE_PATH / "configs"
MODELS_PATH: Path = BASE_PATH / "models"

# Path to the config selector file that stores the filename of the active config file
CONFIG_SELECTOR_PATH: Path = CONFIGS_PATH / "config_selector.yaml"

# Path to the .json file that stores the OAK lens position-temperature mapping for manual focus
MANUAL_FOCUS_PATH: Path = CONFIGS_PATH / "manual_focus.json"

# Name of the file that stores one row of session information per recording session
SESSION_INFO_NAME: str = "session_info.csv"

# Names of the subdirectories in the session directory, which are (optionally)
# created by postprocess.py ('crops', 'overlays') and capture.py ('timelapse')
SUBDIR_NAMES: tuple[str, ...] = ("crops", "overlays", "timelapse")

# Content categories that can be selected for upload and download ('all' excludes overlays)
# all options additionally include the metadata and config files (upload also the log files)
ContentSelection = Literal["all", "full", "crops", "overlays", "timelapse", "metadata"]
CONTENT_OPTIONS: list[str] = list(get_args(ContentSelection))

# Marker files used to coordinate startup sequencing between processes
AUTO_RUN_MARKER: Path = BASE_PATH / ".auto_run_active"
STREAMING_MARKER: Path = BASE_PATH / ".streaming_active"

# OAK image sensor resolution (cropped from full sensor resolution, e.g. 4056x3040 for OAK-1)
SENSOR_RES: tuple[int, int] = (3840, 2160)

# Resolution presets for high-resolution images (capture.py) and streamed frames (webapp.py)
RESOLUTION_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "4k":           (3840, 2160, 1920, 1080),
    "4k-square":    (2176, 2160, 1088, 1080),
    "1080p":        (1920, 1080, 1920, 1080),
    "1080p-square": (1088, 1080, 1088, 1080),
}

# Sensor-space crop dimensions for each resolution preset
SENSOR_CROP: dict[str, tuple[int, int]] = {
    "4k":           (3840, 2160),
    "4k-square":    (2176, 2160),
    "1080p":        (3840, 2160),
    "1080p-square": (2176, 2160),
}

# Usable range of the OAK lens position, and the default value used for manual focus
LENS_POS_MIN: int = 120
LENS_POS_MAX: int = 255
LENS_POS_DEFAULT: int = 156

# Minimum gap (seconds) since the previous recording session before a focus sweep is run
FOCUS_SWEEP_MIN_GAP: int = 1800

# Maximum ISO at which a focus sweep is still attempted (sensor gain adds high-frequency noise)
FOCUS_SWEEP_MAX_ISO: int = 800

# Lens position units for the focus sweep in both directions from the anchor
# Keep this at FOCUS_ANCHOR_MAX_DRIFT + FOCUS_FIT_WINDOW (focus.py)
FOCUS_SWEEP_SPAN: int = 10

# Maximum drift (lens position units) of the sweep anchor from the web app position at equal
# temperature, so a sweep that locks onto the wrong focus plane cannot move the anchor far
FOCUS_ANCHOR_MAX_DRIFT: int = 6

# Maximum time (seconds) between focus corrections, independent of the OAK temperature check interval
FOCUS_CORRECTION_MAX_INTERVAL: int = 30

# Maximum time (seconds) to wait for OAK reconnection before stopping the recording session
OAK_RECONNECT_TIMEOUT: int = 40

# Maximum time (seconds) to keep draining backlog of images to process after recording session ends
IMAGE_DRAIN_TIMEOUT: int = 120

# Available GPIO pins (BCM numbering) for LED (excluding pins that are used by Witty Pi 4 L3V7)
LED_GPIO_PINS: list[int] = [18, 23, 24, 25, 8, 7, 12, 16, 20, 21, 27, 22, 10, 9, 11, 13, 19, 26]

# Minimum password length for Wi-Fi network (WPA2 security standard)
WPA2_PASSWORD_MIN_LENGTH: int = 8

# Maximum time (seconds) to wait for the wlan0 interface to become available at startup
WLAN0_READY_TIMEOUT: int = 30

# Maximum number of Wi-Fi scan attempts before giving up
WIFI_SCAN_ATTEMPTS: int = 3

# Maximum time (seconds) to wait for a Wi-Fi or hotspot connection to be activated
WIFI_CONNECT_TIMEOUT: int = 30

# Average expected download speed (MB/s) over a Raspberry Pi Wi-Fi connection
DOWNLOAD_SPEED: float = 4.5
