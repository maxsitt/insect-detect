from pijuice import PiJuice
import psutil
import time
import subprocess
from pathlib import Path

def check_system_resources(logger):
    # Instantiate PiJuice
    pijuice = PiJuice(1, 0x14)

    # Continue script only if battery charge level and free disk space (MB) are higher than thresholds
    chargelevel_start = pijuice.status.GetChargeLevel().get("data", -1)
    disk_free = round(psutil.disk_usage("/").free / 1048576)
    if chargelevel_start < 10 or disk_free < 200:
        logger.info(f"Shut down without recording | Charge level: {chargelevel_start}%\n")
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
        time.sleep(5) # wait 5 seconds for RPi to shut down

    # Optional: Disable charging of PiJuice battery if charge level is higher than threshold
    #if chargelevel_start > 80:
    #    pijuice.config.SetChargingConfig({"charging_enabled": False})
    return pijuice, chargelevel_start