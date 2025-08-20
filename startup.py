"""Run configured startup sequence at boot (requires enabled systemd service).

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Enable the systemd service 'insect-detect-startup.service' to run this script automatically at boot.

Modify the 'configs/config_selector.yaml' file to select the active configuration file
that will be used to load all configuration parameters.

- write info, warning and error (+ traceback) messages to log file
- load YAML file with configuration parameters
- run startup sequence with optional steps configured in active config file:
  - create RPi Wi-Fi hotspot if it doesn't exist (uses hostname for SSID and password)
  - create/update all configured Wi-Fi profiles in NetworkManager (including hotspot)
  - start primary Python script in new process (if webapp.py: also monitor for user interaction)
  - terminate primary script and start fallback script after the configured delay time
    (and if no user interaction is detected if primary script is webapp.py)
"""

import copy
import logging
import socket
import subprocess
import sys
import time
from gpiozero import LED
from pathlib import Path

from utils.config import parse_yaml, update_config_file
from utils.log import subprocess_log
from utils.network import create_hotspot, set_up_network

# Set base path and get hostname
BASE_PATH = Path.home() / "insect-detect"
HOSTNAME = socket.gethostname()

# Create directory where logs will be stored
LOGS_PATH = BASE_PATH / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Set paths for marker files to indicate web app auto-run and streaming mode
AUTO_RUN_MARKER = BASE_PATH / ".auto_run_active"
STREAMING_MARKER = BASE_PATH / ".streaming_active"  # indicates user interaction with web app

# Set logging levels and format, write logs to file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s",
                    filename=f"{LOGS_PATH}/{Path(__file__).stem}.log", encoding="utf-8")
logger = logging.getLogger()
logger.info("-------- Startup Logger initialized --------")

# Parse active config file and load configuration parameters
config_selector = parse_yaml(BASE_PATH / "configs" / "config_selector.yaml")
config_active = config_selector.config_active
config_active_path = BASE_PATH / "configs" / config_active
config = parse_yaml(config_active_path)
config_updates = copy.deepcopy(dict(config))
logger.info("Configuration %s loaded successfully", config_active)

# Fast-blink LED to indicate startup sequence is running
led = None
if config.led.enabled:
    led_gpio_pin = config.led.gpio_pin
    try:
        led = LED(led_gpio_pin)
    except Exception:
        logger.exception("Error during initialization of LED")
if led:
    led.blink(on_time=0.2, off_time=0.2, background=True)

# Run startup sequence based on active config
if config.startup.hotspot_setup.enabled:
    if config.network.hotspot.ssid is None or config.network.hotspot.password is None:
        logger.info("RPi hotspot SSID and/or password not set, updating both to hostname in config file")
        config_updates["network"]["hotspot"]["ssid"] = HOSTNAME
        config_updates["network"]["hotspot"]["password"] = HOSTNAME
        update_config_file(config_active_path, config_active_path, config_updates, config)
    else:
        logger.info("RPi hotspot SSID and password already set, no changes made to config file")
else:
    logger.info("RPi hotspot setup is disabled, skipping hotspot creation")

if config.startup.network_setup.enabled:
    logger.info("Creating/updating all configured Wi-Fi networks in NetworkManager (including hotspot)")
    try:
        set_up_network(config_updates, activate_network=False)
    except Exception:
        logger.exception("Error during setting up configured Wi-Fi networks")
elif config.startup.hotspot_setup.enabled:
    logger.info("Creating/updating only configured RPi hotspot connection in NetworkManager")
    try:
        create_hotspot(config_updates["network"]["hotspot"])
    except Exception:
        logger.exception("Error during creating/updating configured RPi hotspot connection")
else:
    logger.info("Hotspot and network setup are disabled, skipping creation/update of Wi-Fi networks")

if config.startup.auto_run.enabled:
    logger.info("Auto-run enabled, executing configured Python script(s)")
    if config.startup.auto_run.primary and Path(BASE_PATH / config.startup.auto_run.primary).exists():
        # Create marker file to indicate that auto-run is active (checked by webapp.py)
        AUTO_RUN_MARKER.touch()

        logger.info("Running primary script %s in new process", config.startup.auto_run.primary)
        subprocess_log(LOGS_PATH, config.startup.auto_run.primary)

        with open(LOGS_PATH / "subprocess.log", "a", encoding="utf-8") as log_file_handle:
            primary_process = subprocess.Popen(
                [sys.executable, str(BASE_PATH / config.startup.auto_run.primary)],
                stdout=log_file_handle,
                stderr=log_file_handle,
                start_new_session=True
            )

        if config.startup.auto_run.fallback and Path(BASE_PATH / config.startup.auto_run.fallback).exists():
            delay = config.startup.auto_run.delay
            logger.info("Waiting for %s seconds until primary script is terminated and fallback script is run", delay)
            if config.startup.auto_run.primary == "webapp.py":
                logger.info("Primary script is 'webapp.py', monitoring for user interaction")

            for _ in range(delay):
                time.sleep(1)
                if STREAMING_MARKER.exists():
                    logger.info("User interaction detected, keep web app running without starting fallback script")
                    break
            else:
                logger.info("No user interaction detected for %s seconds, terminating primary script", delay)
                primary_process.terminate()
                try:
                    primary_process.wait(timeout=15)
                    logger.info("Primary script terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Primary script didn't terminate within 15s, force killing")
                    primary_process.kill()
                    primary_process.wait()

                logger.info("Running fallback script %s in new process", config.startup.auto_run.fallback)
                subprocess_log(LOGS_PATH, config.startup.auto_run.fallback)

                with open(LOGS_PATH / "subprocess.log", "a", encoding="utf-8") as log_file_handle:
                    fallback_process = subprocess.Popen(
                        [sys.executable, str(BASE_PATH / config.startup.auto_run.fallback)],
                        stdout=log_file_handle,
                        stderr=log_file_handle,
                        start_new_session=True
                    )
        else:
            logger.info("Fallback script is not specified or does not exist, skipping fallback execution")
    else:
        logger.warning("Primary auto-run script is not specified or does not exist, skipping auto-run")

    # Remove marker files after auto-run is completed
    AUTO_RUN_MARKER.unlink(missing_ok=True)
    STREAMING_MARKER.unlink(missing_ok=True)
else:
    logger.info("Auto-run is disabled, skipping execution of configured Python script(s)")

logger.info("-------- Startup sequence completed --------")
