"""Utility functions to print information or save to log file.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    print_logs(): Print Raspberry Pi information.
    save_logs(): Write information to .csv file during recording.
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


def save_logs(save_path, cam_id, rec_id, device, powermanager=None):
    """Write information to .csv file during recording.

    Write cam ID, recording ID, timestamp, RPi CPU + OAK chip temperature
    and RPi available memory (MB) + CPU utilization (%) to .csv.
    If powermanager (pijuice or wittypi) is provided, also
    write PiJuice or Witty Pi battery info + temp to .csv.
    """
    log_path = save_path.parent / f"{save_path.parent}_info_log.csv"

    try:
        temp_oak = round(device.getChipTemperature().average)
    except RuntimeError:
        temp_oak = "NA"

    try:
        logs = {
            "cam_ID": cam_id,
            "rec_ID": rec_id,
            "timestamp": datetime.now().isoformat(),
            "temp_pi": round(CPUTemperature().temperature),
            "temp_oak": temp_oak,
            "pi_mem_available": round(psutil.virtual_memory().available / 1048576),
            "pi_cpu_used": psutil.cpu_percent(interval=None)
        }
        if powermanager.__class__.__name__ == "PiJuice":
            pijuice = powermanager
            logs.update({
                "power_input": pijuice.status.GetStatus().get("data", {}).get("powerInput", "NA"),
                "charge_status": pijuice.status.GetStatus().get("data", {}).get("battery", "NA"),
                "charge_level": pijuice.status.GetChargeLevel().get("data", 99),
                "voltage_batt_mV": pijuice.status.GetBatteryVoltage().get("data", "NA"),
                "temp_batt": pijuice.status.GetBatteryTemperature().get("data", "NA") 
            })
        elif powermanager.__class__.__name__ == "WittyPiStatus":
            wittypi = powermanager
            logs.update({
                "power_input": wittypi.get_power_mode(),
                "charge_level": wittypi.estimate_chargelevel(),
                "voltage_in_V": wittypi.get_input_voltage(),
                "voltage_out_V": wittypi.get_output_voltage(),
                "current_out_A": wittypi.get_output_current(),
                "temp_wittypi": round(wittypi.get_temperature())
            })
    except IndexError:
        logs = {}

    if logs:
        with open(log_path, "a", encoding="utf-8") as log_file:
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
        metadata = pd.read_csv(metadata_path, encoding="utf-8")
        metadata_tracked = metadata[metadata["track_status"] == "TRACKED"]
        unique_ids = metadata_tracked["track_ID"].nunique()
    except (pd.errors.EmptyDataError, StopIteration):
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

    with open(rec_log_path, "a", encoding="utf-8") as log_rec_file:
        log_rec_writer = csv.DictWriter(log_rec_file, fieldnames=logs_rec.keys())
        if log_rec_file.tell() == 0:
            log_rec_writer.writeheader()
        log_rec_writer.writerow(logs_rec)
