"""Utility functions for network management.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    get_ip_address(): Get the IPv4 address assigned to the wlan0 wireless interface.
    get_current_connection(): Get mode and SSID of the current wifi connection.
    set_up_network(): Set up and activate network connections based on current configuration.
    create_hotspot(): Create or update hotspot connection.
    create_wifi(): Create or update Wi-Fi connections.
"""

import fcntl
import socket
import struct

import nmcli


def get_ip_address():
    """Get the IPv4 address assigned to the wlan0 wireless interface."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Pack interface name into 256-byte buffer (null-padded)
        packed_iface = struct.pack("256s", b"wlan0"[:15])
        # Query kernel for interface address using SIOCGIFADDR (0x8915) ioctl
        packed_addr = fcntl.ioctl(sock.fileno(), 0x8915, packed_iface)
        return socket.inet_ntoa(packed_addr[20:24])  # extract IP address
    except Exception:
        return "127.0.0.1"  # fallback to localhost if wlan0 is not available


def get_current_connection():
    """Get mode and SSID of the current wifi connection."""
    devices = nmcli.device()
    device_wlan0 = next(dev for dev in devices if dev.device == "wlan0")
    current_connection = device_wlan0.connection

    if current_connection and current_connection != "--":
        connection_info = nmcli.connection.show(current_connection)
        connection_ssid = connection_info.get("802-11-wireless.ssid")
        if connection_info.get("802-11-wireless.mode") == "ap":
            return {"mode": "hotspot", "ssid": connection_ssid}
        return {"mode": "wifi", "ssid": connection_ssid}
    return {"mode": "disconnected", "ssid": None}


def set_up_network(config):
    """Set up and activate network connections based on current configuration."""
    devices = nmcli.device()
    device_wlan0 = next(dev for dev in devices if dev.device == "wlan0")
    current_connection = device_wlan0.connection

    highest_priority_wifi = None
    if config["network"]["wifi"]:
        for wifi in config["network"]["wifi"]:
            if wifi.get("ssid") and wifi.get("password"):
                highest_priority_wifi = wifi
                break
        create_wifi(config["network"]["wifi"])

    if config["network"]["hotspot"].get("ssid"):
        create_hotspot(config["network"]["hotspot"])

    if config["network"]["mode"] == "wifi":
        if highest_priority_wifi and current_connection != highest_priority_wifi["ssid"]:
            nmcli.connection.up(str(highest_priority_wifi["ssid"]))
    elif config["network"]["mode"] == "hotspot":
        if current_connection != config["network"]["hotspot"]["ssid"]:
            nmcli.device.disconnect("wlan0")
            nmcli.connection.up(str(config["network"]["hotspot"]["ssid"]))


def create_hotspot(hotspot_config):
    """Create or update hotspot connection."""
    if len(str(hotspot_config["password"])) < 8:
        raise ValueError("Hotspot password must be at least 8 characters long!")

    try:
        nmcli.connection.show(str(hotspot_config["ssid"]))
        connection_exists = True
    except Exception:
        connection_exists = False

    if not connection_exists:
        nmcli.connection.add(
            conn_type="wifi",
            ifname="wlan0",
            name=str(hotspot_config["ssid"]),
            autoconnect=True,
            options={
                "wifi.mode": "ap",
                "wifi.band": "bg",
                "ipv4.method": "shared",
                "ssid": str(hotspot_config["ssid"]),
                "wifi-sec.key-mgmt": "wpa-psk",
                "wifi-sec.psk": str(hotspot_config["password"])
            }
        )
    else:
        nmcli.connection.modify(
            name=str(hotspot_config["ssid"]),
            options={
                "wifi.mode": "ap",
                "wifi.band": "bg",
                "ipv4.method": "shared",
                "wifi-sec.key-mgmt": "wpa-psk",
                "wifi-sec.psk": str(hotspot_config["password"]),
                "connection.autoconnect": "yes"
            }
        )


def create_wifi(wifi_configs):
    """Create or update Wi-Fi connections."""
    for i, wifi_config in enumerate(wifi_configs):
        if not wifi_config.get("ssid") or not wifi_config.get("password"):
            continue

        if len(str(wifi_config["password"])) < 8:
            raise ValueError("Wi-Fi password must be at least 8 characters long!")

        try:
            nmcli.connection.show(str(wifi_config["ssid"]))
            connection_exists = True
        except Exception:
            connection_exists = False

        if not connection_exists:
            nmcli.connection.add(
                conn_type="wifi",
                ifname="wlan0",
                name=str(wifi_config["ssid"]),
                autoconnect=True,
                options={
                    "ssid": str(wifi_config["ssid"]),
                    "wifi-sec.key-mgmt": "wpa-psk",
                    "wifi-sec.psk": str(wifi_config["password"]),
                    "connection.autoconnect-priority": str(100 - (i * 10))
                }
            )
        else:
            nmcli.connection.modify(
                name=str(wifi_config["ssid"]),
                options={
                    "wifi-sec.key-mgmt": "wpa-psk",
                    "wifi-sec.psk": str(wifi_config["password"]),
                    "connection.autoconnect-priority": str(100 - (i * 10)),
                    "connection.autoconnect": "yes"
                }
            )
