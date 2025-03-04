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

partly based on https://github.com/uugear/Witty-Pi-4/blob/main/Software/wittypi/utilities.sh
"""

import logging
import time

from smbus2 import SMBus

# Set I2C bus number
I2C_BUS = 1

# Set I2C device address
I2C_MC_ADDRESS = 0x08

# Set I2C register addresses
I2C_VOLTAGE_IN_I = 1
I2C_VOLTAGE_IN_D = 2
I2C_VOLTAGE_OUT_I = 3
I2C_VOLTAGE_OUT_D = 4
I2C_CURRENT_OUT_I = 5
I2C_CURRENT_OUT_D = 6
I2C_POWER_MODE = 7
I2C_LM75B_TEMPERATURE = 50

# Set minimum and maximum battery voltage (3.7V nominal voltage)
MIN_VOLTAGE = 3.0
MAX_VOLTAGE = 4.2

# Create dictionary that maps voltage to SoC percentage for a typical 3.7V Li-ion battery
VOLTAGE_TO_SOC = {
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
VOLTAGE_KEYS = sorted(VOLTAGE_TO_SOC.keys())


class WittyPiStatus:
    """Read status information from Witty Pi 4 L3V7."""

    def read_i2c_data(self, i2c_address, register_address, read_word=False):
        """Open I2C bus and read data from I2C device."""
        with SMBus(I2C_BUS) as bus:
            try:
                if read_word:
                    i2c_data = bus.read_word_data(i2c_address, register_address)
                else:
                    i2c_data = bus.read_byte_data(i2c_address, register_address)
                return i2c_data
            except Exception:
                return None

    def get_i2c_value(self, int_address, dec_address=None, read_word=False):
        """Get value(s) from the specified I2C register address(es)."""
        i = self.read_i2c_data(I2C_MC_ADDRESS, int_address, read_word)
        d = self.read_i2c_data(I2C_MC_ADDRESS, dec_address, read_word) if dec_address else None

        if i is not None:
            if read_word:
                # Get temperature (°C)
                i = ((i & 0xFF) << 8) | ((i & 0xFF00) >> 8)
                i = i >> 5
                if i >= 0x400:
                    i = (i & 0x3FF) - 1024
                i2c_value = round(i * 0.125, 2)
            elif d is not None:
                # Get voltage (V) or current (A)
                i2c_value = i + d / 100.0
            return i2c_value
        return None

    def get_power_mode(self):
        """Get the power mode (USB C or battery)."""
        power_mode = self.read_i2c_data(I2C_MC_ADDRESS, I2C_POWER_MODE)
        if power_mode is not None:
            if power_mode == 0:
                return "USB_C_IN"
            return "BATTERY_IN"
        return None

    def get_temperature(self):
        """Get the board temperature in °C."""
        return self.get_i2c_value(I2C_LM75B_TEMPERATURE, read_word=True)

    def get_input_voltage(self):
        """Get the input voltage in V."""
        return self.get_i2c_value(I2C_VOLTAGE_IN_I, I2C_VOLTAGE_IN_D)

    def get_output_voltage(self):
        """Get the output voltage in V."""
        return self.get_i2c_value(I2C_VOLTAGE_OUT_I, I2C_VOLTAGE_OUT_D)

    def get_output_current(self):
        """Get the output current in A."""
        return self.get_i2c_value(I2C_CURRENT_OUT_I, I2C_CURRENT_OUT_D)

    def estimate_chargelevel(self):
        """Estimate the battery charge level based on the input voltage."""
        power_mode = self.get_power_mode()
        if power_mode == "USB_C_IN":
            return "USB_C_IN"

        input_voltage = self.get_input_voltage()
        if input_voltage is None:
            return None

        # Linear interpolation between the two closest voltage values
        input_voltage = max(min(input_voltage, MAX_VOLTAGE), MIN_VOLTAGE)
        for i in range(len(VOLTAGE_KEYS) - 1):
            if VOLTAGE_KEYS[i] <= input_voltage <= VOLTAGE_KEYS[i + 1]:
                v1, v2 = VOLTAGE_KEYS[i], VOLTAGE_KEYS[i + 1]
                soc1, soc2 = VOLTAGE_TO_SOC[v1], VOLTAGE_TO_SOC[v2]
                est_chargelevel = int(soc1 + (input_voltage - v1) * (soc2 - soc1) / (v2 - v1))
                return est_chargelevel


def print_info(wittypi):
    """
    Print power mode, temperature, input/output voltage,
    output current and estimated charge level.
    """
    wittypi_info = {
        "Power mode": (wittypi.get_power_mode(), ""),
        "Temperature": (wittypi.get_temperature(), "°C"),
        "Input voltage": (wittypi.get_input_voltage(), "V"),
        "Output voltage": (wittypi.get_output_voltage(), "V"),
        "Output current": (wittypi.get_output_current(), "A"),
        "Estimated charge level": (wittypi.estimate_chargelevel(), "%")
    }

    for key, (get_info, unit) in wittypi_info.items():
        info_value = get_info
        if info_value is not None:
            if key == "Estimated charge level" and info_value == "USB_C_IN":
                logging.info("%s: %s", key, info_value)
            else:
                logging.info("%s: %s %s", key, info_value, unit)
        else:
            logging.info("%s: NA", key)


def main():
    """Print Witty Pi 4 L3V7 status information every 2 seconds."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    wittypi = WittyPiStatus()

    while True:
        print_info(wittypi)
        logging.info("\n")
        time.sleep(2)


if __name__ == "__main__":
    main()
