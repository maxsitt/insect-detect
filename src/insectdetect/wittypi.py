"""Utility class for reading status information from Witty Pi 4 L3V7.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Class:
    WittyPiStatus: Read status information from Witty Pi 4 L3V7.
        Methods:
            read_i2c_data(): Open I2C bus and read data from I2C device.
            get_i2c_value(): Get value(s) from the specified I2C register address(es).
            get_power_mode(): Get the power mode (USB C or battery).
            get_temperature(): Get the board temperature in °C.
            get_input_voltage(): Get the input voltage in V.
            get_output_voltage(): Get the output voltage in V.
            get_output_current(): Get the output current in A.
            estimate_chargelevel(): Estimate the battery charge level based on the input voltage.

Functions:
    print_info(): Print power mode, temperature, input/output voltage,
                  output current and estimated charge level.
    main(): Print Witty Pi 4 L3V7 status information every 2 seconds.

Partly based on https://github.com/uugear/Witty-Pi-4/blob/main/Software/wittypi/utilities.sh
"""

import logging
import time

from smbus2 import SMBus

# I2C bus number and device address
I2C_BUS = 1
I2C_MC_ADDRESS = 0x08

# I2C register addresses
I2C_VOLTAGE_IN_I = 1
I2C_VOLTAGE_IN_D = 2
I2C_VOLTAGE_OUT_I = 3
I2C_VOLTAGE_OUT_D = 4
I2C_CURRENT_OUT_I = 5
I2C_CURRENT_OUT_D = 6
I2C_POWER_MODE = 7
I2C_LM75B_TEMPERATURE = 50

# Minimum and maximum battery voltage (3.7V nominal voltage)
MIN_VOLTAGE = 3.0
MAX_VOLTAGE = 4.2

# Lookup table mapping voltage (V) to state-of-charge (%) for a typical 3.7V Li-ion battery
VOLTAGE_TO_SOC: dict[float, int] = {
    3.0: 0,
    3.3: 10,
    3.4: 20,
    3.5: 30,
    3.6: 40,
    3.7: 50,
    3.8: 60,
    3.9: 70,
    4.0: 80,
    4.1: 90,
    4.2: 100
}
VOLTAGE_KEYS: list[float] = sorted(VOLTAGE_TO_SOC.keys())


class WittyPiStatus:
    """Read status information from Witty Pi 4 L3V7."""

    def read_i2c_data(
        self,
        i2c_address: int,
        register_address: int,
        read_word: bool = False
    ) -> int | None:
        """Open I2C bus and read data from I2C device.

        Args:
            i2c_address:      I2C device address.
            register_address: I2C register address to read from.
            read_word:        If True, read a 16-bit word; otherwise read a single byte.

        Returns:
            Integer value read from the register, or None if the read failed.
        """
        with SMBus(I2C_BUS) as bus:
            try:
                if read_word:
                    return bus.read_word_data(i2c_address, register_address)
                return bus.read_byte_data(i2c_address, register_address)
            except Exception:
                return None

    def get_i2c_value(
        self,
        int_address: int,
        dec_address: int | None = None,
        read_word: bool = False
    ) -> float | None:
        """Get value(s) from the specified I2C register address(es).

        For word reads (temperature), decodes the LM75B two's complement format.
        For byte reads with a decimal register, combines integer and decimal parts
        into a single float representing voltage (V) or current (A).

        Args:
            int_address: I2C register address for the integer part.
            dec_address: I2C register address for the decimal part, or None for word reads.
            read_word:   If True, read a 16-bit word (used for temperature).

        Returns:
            Decoded float value, or None if the read failed.
        """
        i = self.read_i2c_data(I2C_MC_ADDRESS, int_address, read_word)
        d = self.read_i2c_data(I2C_MC_ADDRESS, dec_address) if dec_address is not None else None

        if i is None:
            return None

        if read_word:
            # Decode LM75B two's complement 11-bit temperature value
            i = ((i & 0xFF) << 8) | ((i & 0xFF00) >> 8)
            i = i >> 5
            if i >= 0x400:
                i = (i & 0x3FF) - 1024
            return round(i * 0.125, 2)

        if d is not None:
            # Combine integer and decimal register parts into voltage (V) or current (A)
            return i + d / 100.0

        return None

    def get_power_mode(self) -> str | None:
        """Get the power mode (USB-C or battery).

        Returns:
            'USB_C_IN' if powered via USB-C, 'BATTERY_IN' if on battery,
            or None if the read failed.
        """
        power_mode = self.read_i2c_data(I2C_MC_ADDRESS, I2C_POWER_MODE)
        if power_mode is None:
            return None
        return "USB_C_IN" if power_mode == 0 else "BATTERY_IN"

    def get_temperature(self) -> float | None:
        """Get the board temperature in °C.

        Returns:
            Board temperature as a float in °C, or None if the read failed.
        """
        return self.get_i2c_value(I2C_LM75B_TEMPERATURE, read_word=True)

    def get_input_voltage(self) -> float | None:
        """Get the input voltage in V.

        Returns:
            Input voltage as a float in V, or None if the read failed.
        """
        return self.get_i2c_value(I2C_VOLTAGE_IN_I, I2C_VOLTAGE_IN_D)

    def get_output_voltage(self) -> float | None:
        """Get the output voltage in V.

        Returns:
            Output voltage as a float in V, or None if the read failed.
        """
        return self.get_i2c_value(I2C_VOLTAGE_OUT_I, I2C_VOLTAGE_OUT_D)

    def get_output_current(self) -> float | None:
        """Get the output current in A.

        Returns:
            Output current as a float in A, or None if the read failed.
        """
        return self.get_i2c_value(I2C_CURRENT_OUT_I, I2C_CURRENT_OUT_D)

    def estimate_chargelevel(self) -> int | str | None:
        """Estimate the battery charge level based on the input voltage.

        Uses linear interpolation between the two closest entries in VOLTAGE_TO_SOC.
        Input voltage is clamped to [MIN_VOLTAGE, MAX_VOLTAGE] before interpolation.

        Returns:
            Estimated charge level as an integer percentage (0-100),
            'USB_C_IN' if powered via USB-C (charge level not meaningful),
            or None if the input voltage read failed.
        """
        power_mode = self.get_power_mode()
        if power_mode == "USB_C_IN":
            return "USB_C_IN"

        input_voltage = self.get_input_voltage()
        if input_voltage is None:
            return None

        input_voltage = max(min(input_voltage, MAX_VOLTAGE), MIN_VOLTAGE)
        for i in range(len(VOLTAGE_KEYS) - 1):
            if VOLTAGE_KEYS[i] <= input_voltage <= VOLTAGE_KEYS[i + 1]:
                v1, v2     = VOLTAGE_KEYS[i], VOLTAGE_KEYS[i + 1]
                soc1, soc2 = VOLTAGE_TO_SOC[v1], VOLTAGE_TO_SOC[v2]
                return int(soc1 + (input_voltage - v1) * (soc2 - soc1) / (v2 - v1))

        return None


def print_info(wittypi: WittyPiStatus) -> None:
    """Print power mode, temperature, input/output voltage, output current and charge level.

    Args:
        wittypi: WittyPiStatus instance to read from.
    """
    wittypi_info: dict[str, tuple[int | float | str | None, str]] = {
        "Power mode": (wittypi.get_power_mode(), ""),
        "Temperature": (wittypi.get_temperature(), "°C"),
        "Input voltage": (wittypi.get_input_voltage(), "V"),
        "Output voltage": (wittypi.get_output_voltage(), "V"),
        "Output current": (wittypi.get_output_current(), "A"),
        "Estimated charge level": (wittypi.estimate_chargelevel(), "%"),
    }

    for key, (value, unit) in wittypi_info.items():
        if value is None:
            logging.info("%s: NA", key)
        elif key == "Estimated charge level" and value == "USB_C_IN":
            logging.info("%s: %s", key, value)
        else:
            logging.info("%s: %s %s", key, value, unit)


def main() -> None:
    """Print Witty Pi 4 L3V7 status information every 2 seconds."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    wittypi = WittyPiStatus()

    while True:
        print_info(wittypi)
        logging.info("")
        time.sleep(2)


if __name__ == "__main__":
    main()
