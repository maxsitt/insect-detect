"""Utility functions for network management.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    get_ip_address(): Get the IPv4 address assigned to the wlan0 wireless interface.
    get_current_connection(): Get the mode and SSID of the current Wi-Fi connection.
    set_up_network(): Create and optionally activate Wi-Fi network connections based on config.
    create_hotspot(): Create or update Wi-Fi hotspot connection profile in NetworkManager.
    create_wifi(): Create or update Wi-Fi connection profiles in NetworkManager.
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
    """Get the mode and SSID of the current Wi-Fi connection."""
    devices = nmcli.device()
    device_wlan0 = next(dev for dev in devices if dev.device == "wlan0")
    current_connection = device_wlan0.connection

    if current_connection and current_connection != "--":
        conn_details = nmcli.connection.show(current_connection)
        if conn_details.get("802-11-wireless.mode") == "ap":
            return {"mode": "hotspot", "ssid": conn_details.get("802-11-wireless.ssid")}
        return {"mode": "wifi", "ssid": conn_details.get("802-11-wireless.ssid")}
    return {"mode": "disconnected", "ssid": None}


def set_up_network(config, activate_network=False):
    """Create and optionally activate Wi-Fi network connections based on config."""
    if config["network"]["hotspot"].get("ssid"):
        create_hotspot(config["network"]["hotspot"])

    if config["network"]["wifi"]:
        create_wifi(config["network"]["wifi"])

    if activate_network:
        current_connection = get_current_connection()
        if config["network"]["mode"] == "hotspot":
            if current_connection["ssid"] != config["network"]["hotspot"]["ssid"]:
                nmcli.device.disconnect("wlan0")
                nmcli.connection.up(str(config["network"]["hotspot"]["ssid"]))
        elif config["network"]["mode"] == "wifi":
            highest_priority_wifi = None
            for wifi in config["network"]["wifi"]:
                if wifi.get("ssid") and wifi.get("password"):
                    highest_priority_wifi = wifi
                    break
            if highest_priority_wifi and current_connection["ssid"] != highest_priority_wifi["ssid"]:
                if current_connection["mode"] == "hotspot":
                    nmcli.device.disconnect("wlan0")
                nmcli.connection.up(str(highest_priority_wifi["ssid"]))


def create_hotspot(hotspot_config):
    """Create or update Wi-Fi hotspot connection profile in NetworkManager."""
    if not hotspot_config.get("ssid") or not hotspot_config.get("password"):
        return

    if len(str(hotspot_config["password"])) < 8:
        raise ValueError("Hotspot password must be at least 8 characters long!")

    # Delete any existing hotspot profiles that don't match the configured hotspot SSID
    connections = nmcli.connection()
    for connection in connections:
        if connection.conn_type == "wifi":
            conn_name = connection.name
            conn_details = nmcli.connection.show(conn_name)
            if (conn_details.get("802-11-wireless.mode") == "ap" and
                conn_details.get("802-11-wireless.ssid") != str(hotspot_config["ssid"])):
                nmcli.connection.delete(conn_name)

    try:
        conn_details = nmcli.connection.show(str(hotspot_config["ssid"]), show_secrets=True)
    except Exception:
        conn_details = None

    if not conn_details:
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
        if conn_details.get("802-11-wireless-security.psk") != str(hotspot_config["password"]):
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
    """Create or update Wi-Fi connection profiles in NetworkManager."""
    for i, wifi_config in enumerate(wifi_configs):
        if not wifi_config.get("ssid") or not wifi_config.get("password"):
            continue

        if len(str(wifi_config["password"])) < 8:
            raise ValueError("Wi-Fi password must be at least 8 characters long!")

        # Set priority based on index (= position in list)
        priority = 100 - (i * 10)

        try:
            conn_details = nmcli.connection.show(str(wifi_config["ssid"]), show_secrets=True)
        except Exception:
            conn_details = None

        if not conn_details:
            nmcli.connection.add(
                conn_type="wifi",
                ifname="wlan0",
                name=str(wifi_config["ssid"]),
                autoconnect=True,
                options={
                    "ssid": str(wifi_config["ssid"]),
                    "wifi-sec.key-mgmt": "wpa-psk",
                    "wifi-sec.psk": str(wifi_config["password"]),
                    "connection.autoconnect-priority": str(priority)
                }
            )
        else:
            if (conn_details.get("802-11-wireless-security.psk") != str(wifi_config["password"]) or
                conn_details.get("connection.autoconnect-priority") != str(priority)):
                nmcli.connection.modify(
                    name=str(wifi_config["ssid"]),
                    options={
                        "wifi-sec.key-mgmt": "wpa-psk",
                        "wifi-sec.psk": str(wifi_config["password"]),
                        "connection.autoconnect-priority": str(priority),
                        "connection.autoconnect": "yes"
                    }
                )
