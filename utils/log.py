"""Utility functions for information logging.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    print_logs(): Print Raspberry Pi information.
    save_logs(): Write system information to .csv file during recording.
    record_log(): Write information to .csv file at the end of the recording interval.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
from gpiozero import CPUTemperature


def print_logs():
    """Print Raspberry Pi information."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("\nAvailable RPi memory: %s MB", round(psutil.virtual_memory().available / 1048576))
    logging.info("RPi CPU utilization:  %s %%", round(psutil.cpu_percent(interval=None)))
    logging.info("RPi CPU temperature:  %s °C\n", round(CPUTemperature().temperature))


def save_logs(save_path, cam_id, rec_id, get_temp_oak, get_power_info=None):
    """Write system information to .csv file during recording.

    Write cam ID, recording ID, timestamp, RPi CPU + OAK chip temperature
    and RPi available memory (MB) + CPU utilization (%) to .csv.
    If powermanager (pijuice or wittypi) is enabled, also
    write PiJuice or Witty Pi battery info + temp to .csv.
    """
    log_path = save_path / f"{save_path.name}_system_log.csv"

    try:
        logs = {
            "cam_ID": cam_id,
            "rec_ID": rec_id,
            "timestamp": datetime.now().isoformat(),
            "temp_pi": round(CPUTemperature().temperature),
            "temp_oak": get_temp_oak(),
            "pi_mem_available": round(psutil.virtual_memory().available / 1048576),
            "pi_cpu_used": psutil.cpu_percent(interval=None)
        }
        if get_power_info and callable(get_power_info):
            logs.update(get_power_info())
    except Exception:
        logs = {}

    if logs:
        with open(log_path, "a", buffering=1, encoding="utf-8") as log_file:
            log_writer = csv.DictWriter(log_file, fieldnames=logs.keys())
            if log_file.tell() == 0:
                log_writer.writeheader()
            log_writer.writerow(logs)


def record_log(save_path, cam_id, rec_id, rec_start, rec_end,
               chargelevel_start=None, chargelevel=None):
    """Write information to .csv file at the end of the recording interval.

    Write cam ID, recording ID, recording start and end time, recording duration (min),
    number of unique tracking IDs and available disk space (GB) to .csv.
    If chargelevel_start and chargelevel are provided, also write both to .csv.
    """
    rec_log_path = save_path.parents[1] / "record_log.csv"

    try:
        metadata_path = next(save_path.glob("*metadata.csv"))
        metadata = pd.read_csv(metadata_path, usecols=["track_ID", "track_status"], encoding="utf-8")
        metadata_tracked = metadata[metadata["track_status"] == "TRACKED"]
        unique_ids = metadata_tracked["track_ID"].nunique()
    except Exception:
        unique_ids = 0

    logs_rec = {
        "cam_ID": cam_id,
        "rec_ID": rec_id,
        "rec_start": rec_start.isoformat(),
        "rec_end": rec_end.isoformat(),
        "duration_min": round((rec_end - rec_start).total_seconds() / 60, 2),
        "unique_track_IDs": unique_ids,
        "disk_free_gb": round(psutil.disk_usage("/").free / 1073741824, 1)
    }
    if chargelevel_start is not None and chargelevel is not None:
        logs_rec.update({
            "chargelevel_start": chargelevel_start,
            "chargelevel_end": chargelevel
        })

    with open(rec_log_path, "a", buffering=1, encoding="utf-8") as log_rec_file:
        log_rec_writer = csv.DictWriter(log_rec_file, fieldnames=logs_rec.keys())
        if log_rec_file.tell() == 0:
            log_rec_writer.writeheader()
        log_rec_writer.writerow(logs_rec)
