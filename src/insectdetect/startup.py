"""Run configured startup sequence at boot (requires enabled systemd service).

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Usage:
    Enable the systemd service 'insect-detect-startup.service'
    to run this script automatically after boot.

Startup sequence (optional steps configured in active config file):
    1. Write RPi Wi-Fi hotspot credentials to the config file, derived from the hostname
       if SSID and/or password are not set.
    2. Create/update all configured Wi-Fi profiles in NetworkManager (including hotspot)
       and activate the configured network mode ('wifi' or 'hotspot'). In 'wifi' mode, the
       hotspot is only activated as fallback if no Wi-Fi network can be reached.
    3. Start primary Python script immediately as a new process.
    4. After the configured delay, terminate primary and start fallback script if specified:
        - If primary is 'webapp': fallback is cancelled if a user connected to the live stream
          at any point during the delay window.
        - If primary is 'capture': the full delay always elapses before fallback starts.
"""

import logging
import subprocess
import time
from pathlib import Path

from gpiozero import LED

from insectdetect.config import AppConfig, load_config_selector, load_config_yaml, update_config_yaml
from insectdetect.constants import AUTO_RUN_MARKER, BASE_PATH, HOSTNAME, STREAMING_MARKER, UV, WPA2_PASSWORD_MIN_LENGTH
from insectdetect.metrics import configure_logger, subprocess_log
from insectdetect.network import set_up_network

# Initialize logger for this module
logger = logging.getLogger(__name__)


def _start_led(config: AppConfig) -> LED | None:
    """Initialize and start LED to indicate startup sequence is running.

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
            led.blink(on_time=0.2, off_time=0.2, background=True)  # type: ignore[arg-type]
            return led
        except Exception:
            time.sleep(0.1)
    logger.warning("Could not initialize LED on GPIO pin %s", config.led.gpio_pin)
    return None


def _write_hotspot_credentials(config: AppConfig, config_active_path: Path) -> AppConfig:
    """Write the RPi Wi-Fi hotspot credentials to the active config file.

    If SSID or password are not set in the config, the missing value is derived from the
    hostname and written to the active config file. The config file is re-validated after
    writing and the updated AppConfig is returned, so subsequent steps use the new credentials.

    Skipped if the hotspot is neither set as network mode nor enabled as fallback.

    Args:
        config:             AppConfig with hotspot and startup settings.
        config_active_path: Path to the active config YAML file.

    Returns:
        Updated AppConfig re-validated from the config file after writing new hotspot
        credentials, or the original config unchanged if no updates were needed.
    """
    if not config.network.hotspot_enabled:
        logger.info("RPi hotspot is disabled (no network mode and no fallback), skipping setup")
        return config

    ssid_missing = config.network.hotspot.ssid is None
    password_missing = config.network.hotspot.password is None

    if ssid_missing or password_missing:
        # Only write the missing value, a configured SSID or password is never overwritten
        hotspot_updates: dict[str, object] = {}
        if ssid_missing:
            hotspot_updates["ssid"] = config.network.hotspot.effective_ssid
        if password_missing:
            hotspot_updates["password"] = config.network.hotspot.effective_password

        logger.info(
            "RPi hotspot %s not set, deriving it from hostname '%s' and writing to config file",
            " and ".join(name for name, missing in
                         (("SSID", ssid_missing), ("password", password_missing)) if missing),
            HOSTNAME
        )
        if password_missing and len(HOSTNAME) < WPA2_PASSWORD_MIN_LENGTH:
            logger.info(
                "Hostname is shorter than %s characters, hotspot password padded to '%s'",
                WPA2_PASSWORD_MIN_LENGTH, hotspot_updates["password"]
            )
        return update_config_yaml(config_active_path, {"network": {"hotspot": hotspot_updates}})

    logger.info("RPi hotspot SSID and password already set, no changes made to config file")
    return config


def _configure_network(config: AppConfig) -> None:
    """Set up and activate the configured network, logging failures without aborting startup.

    Args:
        config: AppConfig with startup and network settings.
    """
    logger.info(
        "Setting up Wi-Fi networks in NetworkManager (Wi-Fi sync %s, hotspot %s) "
        "and activating configured network mode '%s'",
        "enabled" if config.startup.wifi_sync.enabled else "disabled",
        "enabled" if config.network.hotspot_enabled else "disabled",
        config.network.mode
    )
    try:
        set_up_network(config, activate_network=True)
    except Exception:
        logger.exception("Failed to set up the configured Wi-Fi networks")


def _run_primary(config: AppConfig, logs_path: Path) -> tuple[subprocess.Popen[bytes], str]:
    """Start the primary script as a subprocess.

    Creates a marker file to indicate that auto-run is active (checked by webapp.py).

    Args:
        config:    AppConfig with auto_run settings.
        logs_path: Directory for subprocess log output.

    Returns:
        Tuple of (process handle, primary script name).
    """
    primary = config.startup.auto_run.primary

    # Create marker file to indicate that auto-run is active (checked by webapp.py)
    AUTO_RUN_MARKER.touch()

    logger.info("Running primary script '%s.py' in new process", primary)
    subprocess_log(f"{primary}.py")

    with open(logs_path / "subprocess.log", "a", encoding="utf-8") as log_file_handle:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            [str(UV), "run", "--no-sync", primary],
            stdout=log_file_handle,
            stderr=log_file_handle,
            cwd=BASE_PATH,
            start_new_session=True
        )

    return process, primary


def _run_fallback(
    config: AppConfig,
    logs_path: Path,
    primary_process: subprocess.Popen[bytes],
    primary: str
) -> None:
    """Wait for delay, then terminate primary and start fallback if no user interaction detected.

    If primary is 'webapp', checks for an active live stream connection once per second
    during the delay - if detected at any point, fallback is cancelled and primary keeps running.
    If primary is 'capture', the streaming marker is never set, so the full delay always elapses.

    Does nothing if fallback is not configured (None).

    Args:
        config:          AppConfig with auto_run settings.
        logs_path:       Directory for subprocess log output.
        primary_process: Subprocess handle for the running primary script.
        primary:         Name of the primary script ('capture' or 'webapp').
    """
    fallback = config.startup.auto_run.fallback
    delay = config.startup.auto_run.delay
    if not fallback:
        logger.info("No fallback script configured, skipping fallback execution")
        return

    if primary == "webapp":
        logger.info(
            "Primary is 'webapp.py': monitoring for live stream connection for %s seconds - "
            "fallback '%s.py' will be cancelled if a user connects",
            delay, fallback
        )
        for _ in range(delay):
            time.sleep(1)
            if STREAMING_MARKER.exists():
                logger.info(
                    "Live stream connection detected, cancelling fallback - "
                    "keeping 'webapp.py' running"
                )
                return
        logger.info("No live stream connection detected within %s seconds", delay)
    else:
        logger.info(
            "Waiting %s seconds before terminating '%s.py' and starting fallback '%s.py'",
            delay, primary, fallback
        )
        time.sleep(delay)

    logger.info("Terminating '%s.py' and starting fallback '%s.py'", primary, fallback)

    primary_process.terminate()
    try:
        primary_process.wait(timeout=15)
        logger.info("'%s.py' terminated gracefully", primary)
    except subprocess.TimeoutExpired:
        logger.warning("'%s.py' did not terminate within 15s, force killing", primary)
        primary_process.kill()
        primary_process.wait()

    logger.info("Running fallback script '%s.py' in new process", fallback)
    subprocess_log(f"{fallback}.py")

    with open(logs_path / "subprocess.log", "a", encoding="utf-8") as log_file_handle:
        subprocess.Popen(
            [str(UV), "run", "--no-sync", fallback],
            stdout=log_file_handle,
            stderr=log_file_handle,
            cwd=BASE_PATH,
            start_new_session=True
        )


def main() -> None:
    """Run configured startup sequence."""
    # Create directory for storing logs and set paths for marker files
    logs_path = BASE_PATH / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)

    # Configure logging (write logs to file)
    configure_logger(Path(__file__).stem)
    logger.info("-------- Startup Logger initialized --------")

    # Parse active configuration file
    config_selector = load_config_selector()
    config_active = config_selector.config_active
    config_active_path = BASE_PATH / "configs" / config_active
    config = load_config_yaml(config_active_path)
    logger.info("Configuration '%s' loaded successfully", config_active)

    # Fast-blink LED to indicate startup sequence is running
    led = _start_led(config)

    # Write the RPi Wi-Fi hotspot credentials to the active config file
    config = _write_hotspot_credentials(config, config_active_path)

    # Create or update Wi-Fi network profiles in NetworkManager and activate network mode
    _configure_network(config)

    # Remove stale marker files from any previous session before starting auto-run
    AUTO_RUN_MARKER.unlink(missing_ok=True)
    STREAMING_MARKER.unlink(missing_ok=True)

    # Start primary and optionally fallback script as subprocess
    if config.startup.auto_run.enabled:
        logger.info("Auto-run enabled, executing configured Python script(s)")
        primary_process, primary = _run_primary(config, logs_path)
        _run_fallback(config, logs_path, primary_process, primary)
    else:
        logger.info("Auto-run is disabled, skipping execution of configured Python script(s)")

    # Remove marker files after startup sequence is completed
    AUTO_RUN_MARKER.unlink(missing_ok=True)
    STREAMING_MARKER.unlink(missing_ok=True)

    # Turn off LED to indicate startup sequence is completed
    if led:
        led.off()

    logger.info("-------- Startup sequence completed --------")


if __name__ == "__main__":
    main()
