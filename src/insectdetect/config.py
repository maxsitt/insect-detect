"""Classes and functions for configuration file management.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    get_field_constraints(): Extract numeric constraints for a nested field from a Pydantic model.
    load_config_selector(): Load the config selector file and return a validated ConfigSelectorModel.
    load_config_yaml(): Load a YAML config file, clamp out-of-range values and return a validated AppConfig.
    check_config_changes(): Return True if two configs differ.
    update_config_selector(): Update the config selector file to point to a different config file.
    update_config_yaml(): Merge updates into the active config file, write back and re-validate.
    sanitize_config(): Return a copy of the config with all passwords masked.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from insectdetect.constants import (CONFIG_SELECTOR_PATH, CONFIGS_PATH, LED_GPIO_PINS,
                                    WPA2_PASSWORD_MIN_LENGTH)

# Initialize logger for this module
logger = logging.getLogger(__name__)

# YAML dump settings for consistent formatting when writing config files
_YAML_DUMP_KWARGS: dict[str, Any] = {
    "default_flow_style": False,
    "allow_unicode": True,
    "sort_keys": False,
    "width": 120,
}

def _float_representer(dumper: yaml.Dumper, value: float) -> yaml.ScalarNode:
    """Represent floats in decimal notation, always keeping at least one decimal place."""
    formatted = f"{value:.6f}".rstrip("0")
    if formatted.endswith("."):
        formatted += "0"  # keep trailing zero to indicate float (e.g. 1.0 instead of 1)
    return dumper.represent_scalar("tag:yaml.org,2002:float", formatted)

# Add custom float representer to ensure consistent formatting of float values in YAML config file
yaml.add_representer(float, _float_representer)


class ConfigSelectorModel(BaseModel):
    """Validated model for the config selector file.

    Stores the filename of the active config file, which must exist in the configs/ directory.
    """
    config_active: str = "config.yaml"


class LocationConfig(BaseModel):
    """Optional GPS coordinates and accuracy for the deployment site."""
    latitude: float | None = None
    longitude: float | None = None
    accuracy: float | None = None


class DeploymentConfig(BaseModel):
    """Optional metadata describing the current field deployment.

    - start: Start time of the camera deployment (ISO 8601 format).
    - location: GPS coordinates and accuracy of the deployment site.
    - setting: Background setting of the camera deployment (e.g. platform/flower species).
    - distance: Distance (in cm) from the camera to the background (e.g. platform/flower).
    - notes: Additional fieldnotes about the deployment.
    """
    start: str | None = None
    location: LocationConfig = LocationConfig()
    setting: str | None = None
    distance: int | None = None
    notes: str | None = None


class ImageConfig(BaseModel):
    """Resolution preset and JPEG quality for captured full images.

    Available resolution presets:
      4k:           3840x2160 image, 1280x720 webapp stream (16:9)
      4k-square:    2176x2160 image,  960x960 webapp stream (1:1)
      1080p:        1920x1080 image, 1280x720 webapp stream (16:9)
      1080p-square: 1088x1080 image,  960x960 webapp stream (1:1)
    """
    resolution: Literal["4k", "4k-square", "1080p", "1080p-square"] = "4k"
    quality: int = Field(default=80, ge=10, le=100)


class ZoomConfig(BaseModel):
    """Crop to zoom factor for restricting the camera field of view if enabled.

    A zoom factor of 1.0 uses the full frame. Values > 1.0 crop symmetrically
    around the center, keeping the original aspect ratio. The crop is applied
    to both the full frame output and the detection model input.
    """
    enabled: bool = False
    factor: float = Field(default=1.0, ge=1.0, le=3.0, multiple_of=0.1)


class FocusManualConfig(BaseModel):
    """Fixed lens position for 'manual' focus mode.

    Two representations of the same position selected by 'focus.type':
    - lens_pos: OAK lens position value (120-255, higher = closer)
    - distance: subject distance in cm (8-80), converted to lens_pos
    """
    lens_pos: int = Field(default=156, ge=120, le=255)
    distance: int = Field(default=19, ge=8, le=80)


class FocusRangeLensPosConfig(BaseModel):
    """Lens position range to restrict auto focus in 'range' mode.

    Lens position values: 120 = infinity, 255 = closest focus distance (8 cm).
    """
    min: int = Field(default=154, ge=120, le=210)
    max: int = Field(default=165, ge=122, le=255)

    @model_validator(mode="after")
    def min_must_be_less_than_max(self) -> "FocusRangeLensPosConfig":
        """Ensure min lens position is strictly less than max lens position."""
        if self.min >= self.max:
            raise ValueError(
                f"focus.range.lens_pos.min ({self.min}) must be less than max ({self.max})"
            )
        return self


class FocusRangeDistanceConfig(BaseModel):
    """Subject distance range (in cm) to restrict auto focus in 'range' mode."""
    min: int = Field(default=15, ge=8, le=75)
    max: int = Field(default=20, ge=9, le=80)

    @model_validator(mode="after")
    def min_must_be_less_than_max(self) -> "FocusRangeDistanceConfig":
        """Ensure min distance is strictly less than max distance."""
        if self.min >= self.max:
            raise ValueError(
                f"focus.range.distance.min ({self.min}) must be less than max ({self.max})"
            )
        return self


class FocusRangeConfig(BaseModel):
    """Auto focus range for 'range' focus mode.

    Two representations of the same ranges selected by 'focus.type':
    - lens_pos: OAK lens position value (120-255, higher = closer)
    - distance: subject distance in cm (8-80), converted to lens_pos
    """
    lens_pos: FocusRangeLensPosConfig = FocusRangeLensPosConfig()
    distance: FocusRangeDistanceConfig = FocusRangeDistanceConfig()


class FocusConfig(BaseModel):
    """Focus settings for the OAK camera.

    Modes:
    - continuous: continuous auto focus (default)
    - manual:     fixed focus position defined in 'focus.manual'
    - range:      auto focus restricted to a range defined in 'focus.range'

    The 'type' field selects whether distances are specified as OAK lens position
    values ('lens_pos') or in cm ('distance') for 'manual' and 'range' modes.
    """
    mode: Literal["continuous", "manual", "range"] = "continuous"
    type: Literal["lens_pos", "distance"] = "lens_pos"
    manual: FocusManualConfig = FocusManualConfig()
    range: FocusRangeConfig = FocusRangeConfig()


class IspConfig(BaseModel):
    """Image Signal Processor (ISP) settings for sharpness and noise reduction."""
    sharpness: int = Field(default=1, ge=0, le=4)
    luma_denoise: int = Field(default=1, ge=0, le=4)
    chroma_denoise: int = Field(default=1, ge=0, le=4)


class CameraConfig(BaseModel):
    """OAK camera settings for frame rate, image quality, zoom, focus and ISP settings."""
    fps: int = Field(default=15, ge=1, le=20)
    image: ImageConfig = ImageConfig()
    zoom: ZoomConfig = ZoomConfig()
    focus: FocusConfig = FocusConfig()
    isp: IspConfig = IspConfig()


class AutoExposureRegionConfig(BaseModel):
    """Use bounding boxes from detections to set the auto exposure region if enabled.

    When enabled, the camera AE algorithm meters exposure from the area
    covered by the most recent detection rather than the full frame.
    """
    enabled: bool = False


class DetectionConfig(BaseModel):
    """Object detection model and inference settings."""
    model: str = "platform_insect-detect-tiled_v2-0-0.tar.xz"
    num_shaves: int = Field(default=4, ge=1, le=10)
    conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    ae_region: AutoExposureRegionConfig = AutoExposureRegionConfig()


class BatteryDurationConfig(BaseModel):
    """Maximum recording durations (in minutes) for each battery charge level.

    Used when 'powermanager.enabled' is true and battery level can be read.
    """
    high: int = Field(default=40, ge=1, le=180)
    medium: int = Field(default=20, ge=1, le=120)
    low: int = Field(default=10, ge=1, le=60)


class RecordingDurationConfig(BaseModel):
    """Maximum recording duration (in minutes) per session.

    'default' is used when power management is disabled or battery level
    cannot be read. 'battery' durations are used otherwise.
    """
    battery: BatteryDurationConfig = BatteryDurationConfig()
    default: int = Field(default=40, ge=1, le=180)


class IntervalConfig(BaseModel):
    """Image capture intervals (in seconds) triggered by detection or timelapse.

    Setting to 0 captures images as fast as possible.
    """
    detection: float = Field(default=1.0, ge=0.0, le=600.0)
    timelapse: float = Field(default=600.0, ge=0.0, le=3600.0)


class ShutdownConfig(BaseModel):
    """Shut down the Raspberry Pi after each recording session if enabled."""
    enabled: bool = True


class RecordingConfig(BaseModel):
    """Settings for recording session duration, capture intervals and shutdown behavior."""
    duration: RecordingDurationConfig = RecordingDurationConfig()
    interval: IntervalConfig = IntervalConfig()
    shutdown: ShutdownConfig = ShutdownConfig()


class CropConfig(BaseModel):
    """Crop individual detections from full images and save as separate files if enabled.

    Methods:
    - square:   crop to a square bounding box - can improve subsequent classification (default)
    - original: crop to the original detection bounding box aspect ratio
    """
    enabled: bool = False
    method: Literal["square", "original"] = "square"


class OverlayConfig(BaseModel):
    """Draw bounding boxes and metadata overlays on full images and save as copies if enabled."""
    enabled: bool = False


class DeleteConfig(BaseModel):
    """Delete original full images after post-processing is complete if enabled.

    Requires at least one of 'crop' or 'overlay' to be enabled,
    so that processed output exists before originals are removed.
    """
    enabled: bool = False


class ProcessingConfig(BaseModel):
    """Post-processing settings applied to full images."""
    crop: CropConfig = CropConfig()
    overlay: OverlayConfig = OverlayConfig()
    delete: DeleteConfig = DeleteConfig()

    @model_validator(mode="after")
    def delete_requires_crop_or_overlay(self) -> "ProcessingConfig":
        """Ensure 'delete' is not enabled unless 'crop' or 'overlay' is also enabled."""
        if self.delete.enabled and not (self.crop.enabled or self.overlay.enabled):
            raise ValueError(
                "'processing.delete' requires 'crop' or 'overlay' to be enabled"
            )
        return self


class StreamConfig(BaseModel):
    """JPEG quality for streamed frames.

    Frame rate and resolution of streamed frames are derived from camera config.
    """
    quality: int = Field(default=70, ge=10, le=100)


class HttpsConfig(BaseModel):
    """Serve the web app over HTTPS if enabled.

    HTTPS is required for the browser Geolocation API to work,
    which is used for automatic GPS coordinate entry in the web app.
    """
    enabled: bool = False


class WebappConfig(BaseModel):
    """Settings for the web app live stream and connection protocol."""
    stream: StreamConfig = StreamConfig()
    https: HttpsConfig = HttpsConfig()


class HotspotConfig(BaseModel):
    """SSID and password for the Raspberry Pi Wi-Fi hotspot."""
    ssid: str | None = None
    password: str | None = None

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str | None) -> str | None:
        """Ensure password meets the minimum length requirement."""
        if v == "":
            return None
        if v is not None and len(v) < WPA2_PASSWORD_MIN_LENGTH:
            raise ValueError(
                f"Hotspot password must be at least {WPA2_PASSWORD_MIN_LENGTH} characters long"
            )
        return v


class WifiNetwork(BaseModel):
    """SSID and password for a Wi-Fi network."""
    ssid: str | None = None
    password: str | None = None

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str | None) -> str | None:
        """Ensure password meets the minimum length requirement."""
        if v == "":
            return None
        if v is not None and len(v) < WPA2_PASSWORD_MIN_LENGTH:
            raise ValueError(
                f"Wi-Fi password must be at least {WPA2_PASSWORD_MIN_LENGTH} characters long"
            )
        return v


class NetworkConfig(BaseModel):
    """Network connection settings.

    Modes:
    - hotspot: start a Raspberry Pi Wi-Fi hotspot for direct device connection
    - wifi:    connect to one of the configured Wi-Fi networks if available (default)
    """
    mode: Literal["hotspot", "wifi"] = "wifi"
    hotspot: HotspotConfig = HotspotConfig()
    wifi: list[WifiNetwork] = [WifiNetwork()]


class PowerManagerConfig(BaseModel):
    """Settings for power management hardware and battery charge monitoring.

    - charge_min: minimum battery charge (%) required to start and continue a recording session
    - charge_check: interval (in seconds) to check the battery charge level during recording
    """
    enabled: bool = True
    model: Literal["wittypi", "pijuice"] = "wittypi"
    charge_min: int = Field(default=30, ge=10, le=95)
    charge_check: int = Field(default=30, ge=1, le=600)


class OakConfig(BaseModel):
    """OAK camera chip temperature monitoring settings.

    - temp_max: maximum allowed chip temperature (°C) before a recording session is stopped
    - temp_check: interval (in seconds) to check the chip temperature during recording
    """
    temp_max: int = Field(default=100, ge=70, le=105)
    temp_check: int = Field(default=30, ge=1, le=600)


class LedConfig(BaseModel):
    """Settings for optional LED control via Raspberry Pi GPIO."""
    enabled: bool = False
    gpio_pin: int = Field(default=18)

    @field_validator("gpio_pin")
    @classmethod
    def validate_gpio_pin(cls, v: int) -> int:
        """Validate that the specified GPIO pin BCM number is in the allowed set."""
        allowed = set(LED_GPIO_PINS)
        if v not in allowed:
            raise ValueError(
                f"led.gpio_pin {v} is not valid. Allowed pins: {sorted(allowed)}"
            )
        return v


class MetricsConfig(BaseModel):
    """Settings for periodic system metrics logging (CPU, RAM, temperature, etc.).

    - interval: interval (in seconds) to log system metrics during recording
    """
    enabled: bool = False
    interval: int = Field(default=30, ge=1, le=600)


class ArchiveConfig(BaseModel):
    """Copy captured data to a separate archive directory if enabled.

    Detection-triggered, timelapse and overlay frames, as well as crops are
    stored in separate uncompressed zip files. Metadata, log and config files
    are copied directly. The oldest original data directories are deleted
    when free disk space drops below the 'disk_low' threshold (in MB).
    """
    enabled: bool = False
    disk_low: int = Field(default=5000, ge=500, le=50000)


class UploadConfig(BaseModel):
    """Upload archived data to a remote server or cloud storage via rclone if enabled.

    Archiving is always run before uploading. The 'content' field selects
    which archived zip files to upload (all options include metadata):
    - all:       upload all data except overlay frames
    - full:      upload only full frame zip files
    - crops:     upload only cropped detection zip files (default)
    - timelapse: upload only timelapse frame zip files
    - metadata:  upload only metadata and log files, no images
    """
    enabled: bool = False
    content: Literal["all", "full", "crops", "timelapse", "metadata"] = "crops"


class StorageConfig(BaseModel):
    """Local storage monitoring, archiving and upload settings.

    - disk_min: minimum required free space (MB) to start and continue a recording session
    - disk_check: interval (in seconds) to check the available disk space during recording
    """
    disk_min: int = Field(default=1000, ge=100, le=10000)
    disk_check: int = Field(default=60, ge=1, le=600)
    archive: ArchiveConfig = ArchiveConfig()
    upload: UploadConfig = UploadConfig()


class HotspotSetupConfig(BaseModel):
    """Automatically create the Raspberry Pi Wi-Fi hotspot on startup if enabled.

    The hotspot is only created if it does not already exist in NetworkManager.
    """
    enabled: bool = True


class NetworkSetupConfig(BaseModel):
    """Automatically configure all Wi-Fi networks from config in NetworkManager if enabled.

    Existing network profiles are updated and new ones are created as needed.
    """
    enabled: bool = True


class AutoRunConfig(BaseModel):
    """Automatically launch 'capture' or 'webapp' on startup if enabled.

    - primary:  application launched immediately on startup
    - fallback: application to launch after 'delay' seconds if 'primary' has not been
                interrupted by user interaction; set to null to disable fallback.
                Only meaningful when different from 'primary'.
    - delay:    time in seconds before 'primary' is terminated and 'fallback' is started.
                If 'primary' is 'webapp', active user interaction (live stream connection)
                detected within this window will cancel the fallback launch.
                If 'primary' is 'capture', the full delay always elapses before fallback starts.
    """
    enabled: bool = False
    primary: Literal["capture", "webapp"] = "capture"
    fallback: Literal["capture", "webapp"] | None = None
    delay: int = Field(default=180, ge=10, le=3600)

    @model_validator(mode="after")
    def fallback_must_differ_from_primary(self) -> "AutoRunConfig":
        """Ensure fallback is not set to the same application as primary."""
        if self.fallback is not None and self.fallback == self.primary:
            raise ValueError(
                f"startup.auto_run.fallback ('{self.fallback}') must differ from "
                f"primary ('{self.primary}')"
            )
        return self


class StartupConfig(BaseModel):
    """Settings for automatic network setup and application launch on startup."""
    hotspot_setup: HotspotSetupConfig = HotspotSetupConfig()
    network_setup: NetworkSetupConfig = NetworkSetupConfig()
    auto_run: AutoRunConfig = AutoRunConfig()


class AppConfig(BaseModel):
    """Validated model containing all insect-detect configuration settings.

    Loaded from a config YAML file. Missing keys are filled with
    default values automatically. Unknown keys are silently ignored.
    Out-of-range numeric values are clamped to their allowed bounds.
    """
    deployment: DeploymentConfig = DeploymentConfig()
    camera: CameraConfig = CameraConfig()
    detection: DetectionConfig = DetectionConfig()
    recording: RecordingConfig = RecordingConfig()
    processing: ProcessingConfig = ProcessingConfig()
    webapp: WebappConfig = WebappConfig()
    network: NetworkConfig = NetworkConfig()
    powermanager: PowerManagerConfig = PowerManagerConfig()
    oak: OakConfig = OakConfig()
    led: LedConfig = LedConfig()
    metrics: MetricsConfig = MetricsConfig()
    storage: StorageConfig = StorageConfig()
    startup: StartupConfig = StartupConfig()


def get_field_constraints(
    model_cls: type[BaseModel],
    *field_path: str
) -> dict[str, int | float | None]:
    """Extract numeric constraints for a nested field from a Pydantic model.

    Traverses the model class hierarchy following the given field path and
    reads ge, le, gt, lt and multiple_of metadata from the final field.

    Args:
        model_cls:   Root Pydantic model class to start traversal from.
        *field_path: Sequence of field name strings forming the path to the
                     target field (e.g. 'camera', 'image', 'width').

    Returns:
        Dict with keys 'min', 'max' and 'multiple_of', all as int/float or None.
        'min' reflects ge (or gt + 1 as fallback), 'max' reflects le (or lt - 1 as fallback).
        All values are None if no constraints are defined or the path is invalid.
    """
    current_cls = model_cls
    field_info = None

    for key in field_path:
        field_info = current_cls.model_fields.get(key)
        if field_info is None:
            return {"min": None, "max": None, "multiple_of": None}
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            current_cls = annotation

    if field_info is None:
        return {"min": None, "max": None, "multiple_of": None}

    ge: int | float | None = None
    le: int | float | None = None
    gt: int | float | None = None
    lt: int | float | None = None
    multiple_of: int | float | None = None
    for meta in field_info.metadata:
        if hasattr(meta, "ge"):
            ge = meta.ge
        if hasattr(meta, "le"):
            le = meta.le
        if hasattr(meta, "gt"):
            gt = meta.gt
        if hasattr(meta, "lt"):
            lt = meta.lt
        if hasattr(meta, "multiple_of"):
            multiple_of = meta.multiple_of

    return {
        "min": ge if ge is not None else (gt + 1 if gt is not None else None),
        "max": le if le is not None else (lt - 1 if lt is not None else None),
        "multiple_of": multiple_of,
    }


def _clamp_raw(
    raw: dict[str, object],
    model_cls: type[BaseModel],
    path: str = ""
) -> tuple[dict[str, object], list[str]]:
    """Recursively clamp numeric values in raw to the bounds defined in model_cls.

    Args:
        raw:       Raw dict loaded from YAML, modified in-place where clamping occurs.
        model_cls: Pydantic model class whose field constraints are used as bounds.
        path:      Dot-separated key path prefix used in correction messages.

    Returns:
        Tuple of (modified raw dict, list of human-readable correction messages).
    """
    corrections: list[str] = []

    for key, field_info in model_cls.model_fields.items():
        if key not in raw or raw[key] is None:
            continue

        value = raw[key]
        full_path = f"{path}.{key}" if path else key
        annotation = field_info.annotation

        # Recurse into nested models
        is_nested_model: bool = (
            isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
            and isinstance(value, dict)
        )
        if is_nested_model:
            raw[key], sub_corrections = _clamp_raw(
                cast(dict[str, object], value),
                cast(type[BaseModel], annotation),
                full_path
            )
            corrections.extend(sub_corrections)
            continue

        # Skip lists (e.g. wifi entries) and non-numeric values
        origin = getattr(annotation, "__origin__", None)
        if origin is list or not isinstance(value, (int, float)):
            continue

        # Extract constraints via get_field_constraints (single-field path from current model)
        constraints = get_field_constraints(model_cls, key)
        lower = constraints["min"]
        upper = constraints["max"]
        multiple_of = constraints["multiple_of"]

        clamped = cast("int | float", value)
        if lower is not None and clamped < lower:
            clamped = lower
        if upper is not None and clamped > upper:
            clamped = upper

        # Snap to multiple_of after clamping, ensure result stays >= lower
        if multiple_of is not None:
            # Use round() to avoid float precision issues (e.g. 1.3 % 0.1 != 0.0 in Python)
            decimal_places = (len(str(multiple_of).rstrip("0").rsplit(".", maxsplit=1)[-1])
                              if "." in str(multiple_of) else 0)
            snapped = round(round(clamped / multiple_of) * multiple_of, decimal_places)
            if snapped != clamped:
                clamped = snapped
            if lower is not None and clamped < lower:
                clamped += multiple_of

        if clamped != value:
            corrections.append(f"  {full_path}: {value!r} -> {clamped!r}")
            raw[key] = clamped

    return raw, corrections


def _deep_update(base: dict[str, object], updates: dict[str, object]) -> None:
    """Recursively merge updates into base dict in-place.

    Nested dicts are merged rather than replaced. All other value
    types (including lists) are overwritten directly.
    """
    for key, value in updates.items():
        base_value = base.get(key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            _deep_update(base_value, value)
            base[key] = base_value
        else:
            base[key] = value


def load_config_selector() -> ConfigSelectorModel:
    """Load config selector file, validate and return a ConfigSelectorModel.

    Raises FileNotFoundError with a list of available config files if
    the referenced config file does not exist in the configs/ directory.

    Returns:
        ConfigSelectorModel with the validated active config filename.
    """
    with open(CONFIG_SELECTOR_PATH, "r", encoding="utf-8") as f:
        raw: dict[str, object] = yaml.safe_load(f) or {}

    selector = ConfigSelectorModel.model_validate(raw)

    config_active_path = CONFIGS_PATH / selector.config_active
    if not config_active_path.exists():
        available = [p.name for p in CONFIGS_PATH.glob("*.yaml")
                     if p.name != CONFIG_SELECTOR_PATH.name]
        raise FileNotFoundError(
            f"Active config file '{selector.config_active}' not found in '{CONFIGS_PATH}'.\n"
            f"Available config files: {available}\n"
            f"Update '{CONFIG_SELECTOR_PATH.name}' to point to an existing file."
        )

    return selector


def load_config_yaml(config_path: Path) -> AppConfig:
    """Load a YAML config file, clamp out-of-range values and return a validated AppConfig.

    Missing keys are filled with Pydantic defaults. Out-of-range numeric values
    are clamped to their allowed bounds and a warning is logged for each correction.

    Args:
        config_path: Path to the active config YAML file.

    Returns:
        Validated AppConfig reflecting the final on-disk state.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, object] = yaml.safe_load(f) or {}

    raw, corrections = _clamp_raw(raw, AppConfig)
    if corrections:
        logger.warning(
            "Config values in '%s' were clamped to their allowed range:\n%s",
            config_path, "\n".join(corrections)
        )

    try:
        config = AppConfig.model_validate(raw)
    except Exception as e:
        raise ValueError(f"Invalid configuration in '{config_path}':\n{e}") from e

    validated_dict = config.model_dump()
    if corrections or raw != validated_dict:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(validated_dict, f, **_YAML_DUMP_KWARGS)

    return config


def check_config_changes(
    original: AppConfig | dict[str, object],
    updates: AppConfig | dict[str, object]
) -> bool:
    """Return True if original and updates represent different configurations.

    Args:
        original: Original configuration (AppConfig or dict).
        updates:  Updated configuration to compare against original (AppConfig or dict).

    Returns:
        True if the configurations differ, False if they are identical.
    """
    original_dict = original if isinstance(original, dict) else original.model_dump()
    updates_dict = updates if isinstance(updates, dict) else updates.model_dump()
    return json.dumps(original_dict, sort_keys=True) != json.dumps(updates_dict, sort_keys=True)


def update_config_selector(config_active: str) -> None:
    """Update the config selector file to point to a different config file.

    Raises FileNotFoundError if the specified config file does not exist
    in the configs/ directory.

    Args:
        config_active: Filename of the config file to set as active (e.g. 'config.yaml').
    """
    config_active_path = CONFIGS_PATH / config_active
    if not config_active_path.exists():
        available = [p.name for p in CONFIGS_PATH.glob("*.yaml")
                     if p.name != CONFIG_SELECTOR_PATH.name]
        raise FileNotFoundError(
            f"Cannot set active config to '{config_active}': file not found in '{CONFIGS_PATH}'.\n"
            f"Available config files: {available}"
        )

    selector = ConfigSelectorModel(config_active=config_active)
    with open(CONFIG_SELECTOR_PATH, "w", encoding="utf-8") as f:
        yaml.dump(selector.model_dump(), f, **_YAML_DUMP_KWARGS)


def update_config_yaml(config_path: Path, config_updates: dict[str, object]) -> AppConfig:
    """Merge updates into a config file, write back and re-validate.

    If the file does not exist (e.g. when creating a new config), an empty base
    dict is used and config_updates is written as-is. Otherwise the current file
    is read first and config_updates is merged into it recursively via _deep_update().
    The result is written back and immediately re-validated, which fills any missing
    keys with Pydantic defaults and clamps out-of-range values.

    Args:
        config_path:    Path to the config YAML file to write (need not exist yet).
        config_updates: Nested dict of changes to apply (e.g. from web app).

    Returns:
        Re-validated AppConfig reflecting the updated file content.
    """
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict[str, object] = yaml.safe_load(f) or {}
    else:
        raw = {}
    _deep_update(raw, config_updates)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, **_YAML_DUMP_KWARGS)

    return load_config_yaml(config_path)


def sanitize_config(config: AppConfig | dict[str, object]) -> dict[str, object]:
    """Return a deep copy of the config with all network passwords masked.

    Args:
        config: Configuration to sanitize (AppConfig or dict).

    Returns:
        Deep-copied dict with all 'password' fields replaced by '[REDACTED]'.
    """
    sanitized: dict[str, object] = copy.deepcopy(
        config if isinstance(config, dict) else config.model_dump()
    )
    network = sanitized.get("network")
    if not isinstance(network, dict):
        return sanitized
    for wifi in network.get("wifi", []):
        if isinstance(wifi, dict) and "password" in wifi:
            wifi["password"] = "[REDACTED]"
    hotspot = network.get("hotspot")
    if isinstance(hotspot, dict) and "password" in hotspot:
        hotspot["password"] = "[REDACTED]"
    return sanitized
