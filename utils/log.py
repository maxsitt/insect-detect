"""Utility functions for information logging.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    get_oak_info(): Get OAK device system information, including CPU and RAM metrics.
    get_rpi_info(): Get Raspberry Pi system information, including CPU and RAM metrics.
    save_logs(): Write system information to .csv file during recording.
    record_log(): Write information to .csv file at the end of the recording interval.
    subprocess_log(): Log information during start of a script executed via subprocess.
"""

import csv
from datetime import datetime

import pandas as pd
import psutil


def get_oak_info(q_syslog):
    """Get OAK device system information, including CPU and RAM metrics."""
    try:
        sysinfo_msg = q_syslog.tryGet()
        if not sysinfo_msg:
            return None

        oak_info = {
            "oak_chip_temp": round(sysinfo_msg.chipTemperature.average, 1),
            "oak_cpu_usage_css": round(sysinfo_msg.leonCssCpuUsage.average * 100, 1),
            "oak_cpu_usage_mss": round(sysinfo_msg.leonMssCpuUsage.average * 100, 1),
            "oak_ram_usage_ddr": round((sysinfo_msg.ddrMemoryUsage.used / sysinfo_msg.ddrMemoryUsage.total) * 100, 1),
            "oak_ram_available_ddr": round(sysinfo_msg.ddrMemoryUsage.remaining / 1048576, 1),
            "oak_ram_usage_css": round((sysinfo_msg.leonCssMemoryUsage.used / sysinfo_msg.leonCssMemoryUsage.total) * 100, 1),
            "oak_ram_available_css": round(sysinfo_msg.leonCssMemoryUsage.remaining / 1048576, 1),
            "oak_ram_usage_mss": round((sysinfo_msg.leonMssMemoryUsage.used / sysinfo_msg.leonMssMemoryUsage.total) * 100, 1),
            "oak_ram_available_mss": round(sysinfo_msg.leonMssMemoryUsage.remaining / 1048576, 1),
            "oak_ram_usage_cmx": round((sysinfo_msg.cmxMemoryUsage.used / sysinfo_msg.cmxMemoryUsage.total) * 100, 1),
            "oak_ram_available_cmx": round(sysinfo_msg.cmxMemoryUsage.remaining / 1048576, 2)
        }

        return oak_info
    except Exception:
        return None


def get_rpi_info():
    """Get Raspberry Pi system information, including CPU and RAM metrics."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as f:
            cpu_temp = float(f.read().strip()) / 1000
        cpu_usage_per_core = psutil.cpu_percent(interval=None, percpu=True)
        cpu_usage_sum = sum(cpu_usage_per_core)
        cpu_usage_avg = cpu_usage_sum / len(cpu_usage_per_core)
        ram = psutil.virtual_memory()

        rpi_info = {
            "rpi_cpu_temp": round(cpu_temp, 1),
            "rpi_cpu_usage_avg": round(cpu_usage_avg, 1),  # average CPU usage of all cores (0-100%)
            "rpi_cpu_usage_sum": round(cpu_usage_sum, 1),  # CPU usage sum of all cores (can be >100%)
            "rpi_ram_usage": ram.percent,
            "rpi_ram_available": round(ram.available / 1048576)
        }

        return rpi_info
    except Exception:
        return None


def save_logs(save_path, cam_id, rec_id, q_syslog, get_power_info=None):
    """Write system information to .csv file during recording."""
    log_path = save_path / f"{save_path.name}_system_log.csv"

    try:
        timestamp = datetime.now().isoformat()
        rpi_info = get_rpi_info()
        oak_info = get_oak_info(q_syslog)

        logs = {
            "cam_ID": cam_id,
            "rec_ID": rec_id,
            "timestamp": timestamp
        }

        if rpi_info:
            logs.update(rpi_info)
        if oak_info:
            logs.update(oak_info)
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


def record_log(save_path, cam_id, rec_id, rec_start, rec_end, chargelevel_start=None, chargelevel=None):
    """Write information to .csv file at the end of the recording interval."""
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


def subprocess_log(logs_path, script_name):
    """Log information during start of a script executed via subprocess."""
    timestamp = datetime.now().strftime("%F %T")
    with open(logs_path / "subprocess.log", "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - Running {script_name} via subprocess\n")
