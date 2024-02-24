import csv
import time
from datetime import datetime
import psutil
from pathlib import Path
from pijuice import PiJuice  # If using PiJuice in this context
from gpiozero import CPUTemperature
import pandas as pd

def record_log(rec_id, rec_start, save_path, chargelevel_start, chargelevel, start_time):
    """Write information about each recording interval to .csv file."""
    try:
        df_meta = pd.read_csv(f"{save_path}/metadata_{rec_start}.csv", encoding="utf-8")
        unique_ids = df_meta["track_ID"].nunique()
    except pd.errors.EmptyDataError:
        unique_ids = 0
    with open(f"{save_path}/record_log.csv", "a", encoding="utf-8") as log_rec_file:
        log_rec = csv.DictWriter(log_rec_file, fieldnames=[
            "rec_ID", "record_start_date", "record_start_time", "record_end_time", "record_time_min",
            "num_crops", "num_IDs", "disk_free_gb", "chargelevel_start", "chargelevel_end"
        ])
        if log_rec_file.tell() == 0:
            log_rec.writeheader()
        logs_rec = {
            "rec_ID": rec_id,
            "record_start_date": rec_start[:8],
            "record_start_time": rec_start[9:],
            "record_end_time": datetime.now().strftime("%H-%M"),
            "record_time_min": round((time.monotonic() - start_time) / 60, 2),
            "num_crops": len(list(Path(f"{save_path}/cropped").glob("**/*.jpg"))),
            "num_IDs": unique_ids,
            "disk_free_gb": round(psutil.disk_usage("/").free / 1073741824, 1),
            "chargelevel_start": chargelevel_start,
            "chargelevel_end": chargelevel  # This needs to be defined or passed to the function
        }
        log_rec.writerow(logs_rec)


def save_logs(rec_id, rec_start, chargelevel, pijuice, device):
    """Write recording ID, time, RPi CPU + OAK chip temperature, RPi available memory (MB) +
    CPU utilization (%) and PiJuice battery info + temp to .csv file."""
    with open(f"insect-detect/data/{rec_start[:8]}/info_log_{rec_start[:8]}.csv", "a", encoding="utf-8") as log_info_file:
        log_info = csv.DictWriter(log_info_file, fieldnames=[
            "rec_ID", "timestamp", "temp_pi", "temp_oak", "pi_mem_available", "pi_cpu_used",
            "power_input", "charge_status", "charge_level", "temp_batt", "voltage_batt_mV"
        ])
        if log_info_file.tell() == 0:
            log_info.writeheader()
        try:
            temp_oak = round(device.getChipTemperature().average)  # This might need to be adjusted based on actual method to get temperature
        except RuntimeError:
            temp_oak = "NA"
        try:
            logs_info = {
                "rec_ID": rec_id,
                "timestamp": datetime.now().strftime("%Y%m%d_%H-%M-%S"),
                "temp_pi": round(CPUTemperature().temperature),
                "temp_oak": temp_oak,
                "pi_mem_available": round(psutil.virtual_memory().available / 1048576),
                "pi_cpu_used": psutil.cpu_percent(interval=None),
                "power_input": pijuice.status.GetStatus().get("data", {}).get("powerInput", "NA"),
                "charge_status": pijuice.status.GetStatus().get("data", {}).get("battery", "NA"),
                "charge_level": chargelevel,
                "temp_batt": pijuice.status.GetBatteryTemperature().get("data", "NA"),
                "voltage_batt_mV": pijuice.status.GetBatteryVoltage().get("data", "NA")
            }
        except IndexError:
            logs_info = {}
        log_info.writerow(logs_info)
        log_info_file.flush()
