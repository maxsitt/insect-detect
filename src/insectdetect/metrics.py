"""Utility functions for system metrics and information logging.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    configure_logger(): Configure root logger to write to file, optionally also to stdout.
    get_oak_metrics(): Get OAK camera system metrics, including temperature and CPU + RAM usage.
    get_rpi_metrics(): Get Raspberry Pi system metrics, including temperature and CPU + RAM usage.
    save_metrics(): Write system metrics to .csv file during the recording session.
    save_session_info(): Write session information to .csv file at the end of the recording session.
    subprocess_log(): Log information during start of a script executed via subprocess.
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, cast

import depthai as dai
import polars as pl
import psutil

from insectdetect.constants import LOGS_PATH


def configure_logger(script_name: str, stream_to_console: bool = False) -> None:
    """Configure root logger to write to file, optionally also to stdout.

    Args:
        script_name:       Name of the script, used as the log filename stem.
        stream_to_console: If True, also stream log output to stdout.
    """
    handlers: list[logging.Handler] = [
        logging.FileHandler(LOGS_PATH / f"{script_name}.log", encoding="utf-8")
    ]
    if stream_to_console:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=handlers
    )


def get_oak_metrics(q_syslog: dai.MessageQueue) -> dict[str, object] | None:
    """Get OAK camera system metrics, including temperature and CPU + RAM usage.

    Args:
        q_syslog: depthai output queue for SystemInformation messages.

    Returns:
        Dict of OAK system metrics, or None if no message is available or an error occurs.
    """
    try:
        sysinfo = cast(dai.SystemInformation | None, q_syslog.tryGet())
        if sysinfo is None:
            return None

        return {
            "oak_chip_temp": round(sysinfo.chipTemperature.average, 1),
            "oak_cpu_usage_css": round(sysinfo.leonCssCpuUsage.average * 100, 1),
            "oak_cpu_usage_mss": round(sysinfo.leonMssCpuUsage.average * 100, 1),
            "oak_ram_usage_ddr": round((sysinfo.ddrMemoryUsage.used / sysinfo.ddrMemoryUsage.total) * 100, 1),
            "oak_ram_available_ddr": round(sysinfo.ddrMemoryUsage.remaining / 1048576, 1),
            "oak_ram_usage_css": round((sysinfo.leonCssMemoryUsage.used / sysinfo.leonCssMemoryUsage.total) * 100, 1),
            "oak_ram_available_css": round(sysinfo.leonCssMemoryUsage.remaining / 1048576, 1),
            "oak_ram_usage_mss": round((sysinfo.leonMssMemoryUsage.used / sysinfo.leonMssMemoryUsage.total) * 100, 1),
            "oak_ram_available_mss": round(sysinfo.leonMssMemoryUsage.remaining / 1048576, 1),
            "oak_ram_usage_cmx": round((sysinfo.cmxMemoryUsage.used / sysinfo.cmxMemoryUsage.total) * 100, 1),
            "oak_ram_available_cmx": round(sysinfo.cmxMemoryUsage.remaining / 1048576, 2),
        }
    except Exception:
        return None


def get_rpi_metrics() -> dict[str, object] | None:
    """Get Raspberry Pi system metrics, including temperature and CPU + RAM usage.

    Returns:
        Dict of Raspberry Pi system metrics, or None if an error occurs.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as f:
            cpu_temp = float(f.read().strip()) / 1000
        cpu_usage_per_core: list[float] = psutil.cpu_percent(interval=None, percpu=True)
        cpu_usage_sum = sum(cpu_usage_per_core)
        cpu_usage_avg = cpu_usage_sum / len(cpu_usage_per_core)
        ram = psutil.virtual_memory()

        return {
            "rpi_cpu_temp":      round(cpu_temp, 1),
            "rpi_cpu_usage_avg": round(cpu_usage_avg, 1),  # average CPU usage of all cores (0-100%)
            "rpi_cpu_usage_sum": round(cpu_usage_sum, 1),  # CPU usage sum of all cores (can be >100%)
            "rpi_ram_usage":     ram.percent,
            "rpi_ram_available": round(ram.available / 1048576),
        }
    except Exception:
        return None


def save_metrics(
    session_path: Path,
    device_id: str,
    session_id: int,
    q_syslog: dai.MessageQueue,
    get_power_info: Callable[[], dict[str, object]] | None = None
) -> None:
    """Write system metrics to .csv file during a recording session.

    Args:
        session_path:   Recording session directory where the log file is written.
        device_id:      Camera trap ID (hostname).
        session_id:     Incrementing recording session counter.
        q_syslog:       depthai output queue for SystemInformation messages.
        get_power_info: Optional callable returning a dict of power metrics for logging.
    """
    try:
        timestamp = datetime.now().isoformat()
        rpi_info = get_rpi_metrics()
        oak_info = get_oak_metrics(q_syslog)

        logs: dict[str, object] = {
            "device_id":  device_id,
            "session_id": session_id,
            "timestamp":  timestamp,
        }
        if rpi_info:
            logs.update(rpi_info)
        if oak_info:
            logs.update(oak_info)
        if get_power_info is not None:
            logs.update(get_power_info())
    except Exception:
        logs = {}

    if logs:
        system_log_path = session_path / f"{session_path.name}_system_log.csv"
        with open(system_log_path, "a", buffering=1, encoding="utf-8") as log_file:
            log_writer = csv.DictWriter(log_file, fieldnames=logs.keys())
            if log_file.tell() == 0:
                log_writer.writeheader()
            log_writer.writerow(logs)


def save_session_info(
    data_path: Path,
    session_path: Path,
    device_id: str,
    session_id: int,
    session_start: datetime,
    session_end: datetime,
    chargelevel_start: int | str | None = None,
    chargelevel_end: int | str | None = None
) -> None:
    """Write session information to .csv file at the end of the recording session.

    Appends one row per session to a shared session_info.csv in the data directory.

    Args:
        session_path:      Recording session directory.
        device_id:         Camera trap ID (hostname).
        session_id:        Incrementing recording session counter.
        session_start:     Session start datetime.
        session_end:       Session end datetime.
        chargelevel_start: Battery charge level at session start (% or 'USB_C_IN' or 'NA').
        chargelevel:       Battery charge level at session end (% or 'USB_C_IN' or 'NA').
    """
    session_info_path = data_path / "session_info.csv"
    try:
        metadata_path = next(session_path.glob("*metadata.csv"))
        metadata = pl.read_csv(metadata_path, columns=["track_id", "track_status"])
        unique_ids: int = (
            metadata.filter(pl.col("track_status") == "TRACKED")["track_id"].n_unique()
        )
    except Exception:
        unique_ids = 0

    session_info: dict[str, object] = {
        "device_id": device_id,
        "session_id": session_id,
        "session_start": session_start.isoformat(),
        "session_end": session_end.isoformat(),
        "duration_min": round((session_end - session_start).total_seconds() / 60, 2),
        "unique_track_ids": unique_ids,
        "disk_free_gb": round(psutil.disk_usage("/").free / 1073741824, 1),
    }
    if chargelevel_start is not None and chargelevel_end is not None:
        session_info["chargelevel_start"] = chargelevel_start
        session_info["chargelevel_end"] = chargelevel_end

    with open(session_info_path, "a", buffering=1, encoding="utf-8") as session_info_file:
        session_info_writer = csv.DictWriter(session_info_file, fieldnames=session_info.keys())
        if session_info_file.tell() == 0:
            session_info_writer.writeheader()
        session_info_writer.writerow(session_info)


def subprocess_log(script_name: str) -> None:
    """Log information during start of a script executed via subprocess.

    Appends a timestamped line to a shared subprocess.log file.

    Args:
        script_name: Name of the script being launched via subprocess.
    """
    timestamp = datetime.now().strftime("%F %T")
    with open(LOGS_PATH / "subprocess.log", "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - Running {script_name} via subprocess\n")
