"""Shared constants used across insect-detect modules.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/
"""

import socket
from pathlib import Path

# Hostname used as device identifier (camera trap ID)
HOSTNAME: str = socket.gethostname()

# Path to uv executable
UV: Path = Path.home() / ".local" / "bin" / "uv"

# Paths to root directory and subdirectories for data, logs, configs, and models
BASE_PATH: Path = Path.home() / "insect-detect"
DATA_PATH: Path = BASE_PATH / "data"
LOGS_PATH: Path = BASE_PATH / "logs"
CONFIGS_PATH: Path = BASE_PATH / "configs"
MODELS_PATH: Path = BASE_PATH / "models"

# Path to the config selector file that stores the filename of the active config file
CONFIG_SELECTOR_PATH: Path = CONFIGS_PATH / "config_selector.yaml"

# Marker files used to coordinate startup sequencing between processes
AUTO_RUN_MARKER: Path = BASE_PATH / ".auto_run_active"
STREAMING_MARKER: Path = BASE_PATH / ".streaming_active"

# OAK image sensor resolution (cropped from full sensor resolution, e.g. 4056x3040 for OAK-1)
SENSOR_RES = (3840, 2160)

# Resolution presets for high-resolution images (capture.py) and streamed frames (webapp.py)
RESOLUTION_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "4k":           (3840, 2160, 1280, 720),
    "4k-square":    (2176, 2160,  960, 960),
    "1080p":        (1920, 1080, 1280, 720),
    "1080p-square": (1088, 1080,  960, 960),
}

# Sensor-space crop dimensions for each resolution preset
SENSOR_CROP: dict[str, tuple[int, int]] = {
    "4k":           (3840, 2160),
    "4k-square":    (2176, 2160),
    "1080p":        (3840, 2160),
    "1080p-square": (2176, 2160),
}

# Available GPIO pins (BCM numbering) for LED (excluding pins that are used by Witty Pi 4 L3V7)
LED_GPIO_PINS: list[int] = [18, 23, 24, 25, 8, 7, 12, 16, 20, 21, 27, 22, 10, 9, 11, 13, 19, 26]

# Minimum password length for Wi-Fi network (WPA2 security standard)
WPA2_PASSWORD_MIN_LENGTH: int = 8
