"""Utility functions for power management.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    create_signal_handler(): Create signal handler for a received signal.
    init_power_manager(): Initialize power manager (Witty Pi 4 L3V7 or PiJuice).
"""

import signal
import threading
import time


def create_signal_handler(external_shutdown):
    """Create signal handler for a received signal."""

    def signal_handler(sig, frame):
        """Handle a received signal by setting an external shutdown event."""
        external_shutdown.set()

    return signal_handler


def init_power_manager(power_manager_model):
    """Initialize power manager (Witty Pi 4 L3V7 or PiJuice)."""
    charge_lock = threading.Lock()  # make access to charge level thread-safe

    if power_manager_model == "wittypi":
        from utils.wittypi import WittyPiStatus

        wittypi = WittyPiStatus()

        test_voltage = wittypi.get_input_voltage()
        if test_voltage is None:
            raise RuntimeError("Failed to communicate with Witty Pi board")

        def get_chargelevel():
            """Get battery charge level from Witty Pi 4 L3V7 with retries."""
            with charge_lock:
                for attempt in range(5):
                    result = wittypi.estimate_chargelevel()
                    if result is not None:
                        return result
                    time.sleep(0.01)
            return "NA"

        def get_power_info():
            """Get information from Witty Pi 4 L3V7 for logging."""
            return {
                "power_input": wittypi.get_power_mode(),
                "charge_level": get_chargelevel(),
                "voltage_in_V": wittypi.get_input_voltage(),
                "voltage_out_V": wittypi.get_output_voltage(),
                "current_out_A": wittypi.get_output_current(),
                "temp_wittypi": round(wittypi.get_temperature(), 1)
            }

        # Handle SIGTERM signal (e.g. from button as external shutdown trigger)
        external_shutdown = threading.Event()
        signal.signal(signal.SIGTERM, create_signal_handler(external_shutdown))

        return get_chargelevel, get_power_info, external_shutdown

    elif power_manager_model == "pijuice":
        from pijuice import PiJuice

        pijuice = PiJuice(1, 0x14)

        def get_chargelevel():
            """Get battery charge level from PiJuice with retries."""
            with charge_lock:
                for attempt in range(5):
                    result = pijuice.status.GetChargeLevel()
                    if "data" in result:
                        return result["data"]
                    time.sleep(0.01)
            return "NA"

        def get_power_info():
            """Get information from PiJuice for logging."""
            return {
                "power_input": pijuice.status.GetStatus().get("data", {}).get("powerInput", "NA"),
                "charge_status": pijuice.status.GetStatus().get("data", {}).get("battery", "NA"),
                "charge_level": get_chargelevel(),
                "voltage_batt_mV": pijuice.status.GetBatteryVoltage().get("data", "NA"),
                "temp_batt": pijuice.status.GetBatteryTemperature().get("data", "NA")
            }

        # Create external shutdown dummy event
        external_shutdown = threading.Event()

        return get_chargelevel, get_power_info, external_shutdown

    else:
        # Return dummy functions and event if no valid power manager is specified
        return lambda: None, lambda: {}, threading.Event()
