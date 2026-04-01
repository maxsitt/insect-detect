"""Utility functions for power management initialization and state handling.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Classes:
    PowerManagerState: Bundles all power management state and callbacks into a single object.

Functions:
    create_signal_handler(): Create signal handler for a received signal.
    init_power_manager(): Initialize power manager and return a PowerManagerState.
"""

import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType


@dataclass
class PowerManagerState:
    """Bundled power management state and callbacks.

    When disabled, use PowerManagerState.disabled() to get a safe default instance.

    Attributes:
        enabled:           True if power management is active and initialized.
        chargelevel_start: Battery charge level at session start (% or 'USB_C_IN' or 'NA').
        get_chargelevel:   Callable that returns the current battery charge level.
        get_power_info:    Callable that returns a dict of power-related metrics for logging.
        external_shutdown: Event set by signal handler (e.g. button press) to stop recording.
    """
    enabled: bool
    chargelevel_start: int | str | None
    get_chargelevel: Callable[[], int | str | None]
    get_power_info: Callable[[], dict[str, object]]
    external_shutdown: threading.Event

    @classmethod
    def disabled(cls) -> "PowerManagerState":
        """Return a safe default PowerManagerState with power management disabled."""
        return cls(
            enabled=False,
            chargelevel_start=None,
            get_chargelevel=lambda: None,
            get_power_info=lambda: {},
            external_shutdown=threading.Event()
        )


def create_signal_handler(
    external_shutdown: threading.Event
) -> Callable[[int, FrameType | None], None]:
    """Create signal handler for a received signal.

    Args:
        external_shutdown: Event to set when the signal is received.

    Returns:
        Signal handler function that sets the external_shutdown event.
    """

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        """Handle a received signal by setting an external shutdown event.

        Args:
            signum: Signal number received.
            frame:  Current stack frame at the time of the signal (may be None).
        """
        external_shutdown.set()

    return signal_handler


def init_power_manager(power_manager_model: str | None) -> PowerManagerState:
    """Initialize power manager (Witty Pi 4 L3V7 or PiJuice) and return a PowerManagerState.

    Args:
        power_manager_model: Power manager model identifier ('wittypi' or 'pijuice'),
                             or None to disable power management.

    Returns:
        PowerManagerState with all power management state and callbacks.

    Raises:
        RuntimeError: If communication with the Witty Pi board fails.
    """
    charge_lock = threading.Lock()  # make access to charge level thread-safe

    if power_manager_model == "wittypi":
        from insectdetect.wittypi import WittyPiStatus

        wittypi = WittyPiStatus()

        if wittypi.get_input_voltage() is None:
            raise RuntimeError("Failed to communicate with Witty Pi board")

        def get_chargelevel() -> int | str | None:
            """Get battery charge level from Witty Pi 4 L3V7 with retries."""
            with charge_lock:
                for _ in range(5):
                    result = wittypi.estimate_chargelevel()
                    if result is not None:
                        return result
                    time.sleep(0.01)
            return "NA"

        def get_power_info() -> dict[str, object]:
            """Get information from Witty Pi 4 L3V7 for logging."""
            temp = wittypi.get_temperature()
            return {
                "power_input": wittypi.get_power_mode(),
                "charge_level": get_chargelevel(),
                "voltage_in_V": wittypi.get_input_voltage(),
                "voltage_out_V": wittypi.get_output_voltage(),
                "current_out_A": wittypi.get_output_current(),
                "temp_wittypi": round(temp, 1) if temp is not None else None,
            }

        external_shutdown = threading.Event()
        signal.signal(signal.SIGTERM, create_signal_handler(external_shutdown))

        return PowerManagerState(
            enabled=True,
            chargelevel_start=get_chargelevel(),
            get_chargelevel=get_chargelevel,
            get_power_info=get_power_info,
            external_shutdown=external_shutdown
        )

    elif power_manager_model == "pijuice":
        from pijuice import PiJuice

        pijuice = PiJuice(1, 0x14)

        def get_chargelevel() -> int | str | None:
            """Get battery charge level from PiJuice with retries."""
            with charge_lock:
                for _ in range(5):
                    result = pijuice.status.GetChargeLevel()
                    if "data" in result:
                        return result["data"]
                    time.sleep(0.01)
            return "NA"

        def get_power_info() -> dict[str, object]:
            """Get information from PiJuice for logging."""
            status = pijuice.status.GetStatus().get("data", {})
            return {
                "power_input": status.get("powerInput", "NA"),
                "charge_status": status.get("battery", "NA"),
                "charge_level": get_chargelevel(),
                "voltage_batt_mV": pijuice.status.GetBatteryVoltage().get("data", "NA"),
                "temp_batt": pijuice.status.GetBatteryTemperature().get("data", "NA"),
            }

        return PowerManagerState(
            enabled=True,
            chargelevel_start=get_chargelevel(),
            get_chargelevel=get_chargelevel,
            get_power_info=get_power_info,
            external_shutdown=threading.Event(),
        )

    else:
        return PowerManagerState.disabled()
