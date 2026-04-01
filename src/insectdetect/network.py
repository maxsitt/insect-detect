"""Utility functions for network management via NetworkManager.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    get_ip_address(): Get the IPv4 address assigned to the wlan0 wireless interface.
    get_current_connection(): Get the mode and SSID of the current Wi-Fi connection.
    get_visible_ssids(): Return a set of SSIDs currently visible in a Wi-Fi scan.
    connect_wifi_with_fallback(): Try to connect to configured Wi-Fi network.
    set_up_network(): Create and optionally activate Wi-Fi network connections based on config.
    create_hotspot(): Create or update Wi-Fi hotspot connection profile in NetworkManager.
    create_wifi(): Create or update Wi-Fi connection profiles in NetworkManager.
"""

import fcntl
import logging
import socket
import struct
from typing import TypedDict

import nmcli

from insectdetect.config import AppConfig

# Initialize logger for this module
logger = logging.getLogger(__name__)


class ConnectionInfo(TypedDict):
    """Current Wi-Fi connection mode and SSID."""
    mode: str
    ssid: str | None


def get_ip_address() -> str:
    """Get the IPv4 address assigned to the wlan0 wireless interface.

    Returns:
        IPv4 address string, or '127.0.0.1' if wlan0 is not available.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Pack interface name into 256-byte buffer (null-padded)
        packed_iface = struct.pack("256s", b"wlan0"[:15])
        # Query kernel for interface address using SIOCGIFADDR (0x8915) ioctl
        packed_addr = fcntl.ioctl(sock.fileno(), 0x8915, packed_iface)
        return socket.inet_ntoa(packed_addr[20:24])  # extract IP address
    except Exception:
        return "127.0.0.1"  # fallback to localhost if wlan0 is not available


def get_current_connection() -> ConnectionInfo:
    """Get the mode and SSID of the current Wi-Fi connection.

    Returns:
        ConnectionInfo with 'mode' ('hotspot', 'wifi' or 'disconnected')
        and 'ssid' (str or None).
    """
    try:
        devices = nmcli.device()
        device_wlan0 = next((dev for dev in devices if dev.device == "wlan0"), None)
        if device_wlan0 is None:
            logger.warning("wlan0 interface not found.")
            return ConnectionInfo(mode="disconnected", ssid=None)

        current_connection = device_wlan0.connection
        if current_connection and current_connection != "--":
            conn_details = nmcli.connection.show(current_connection)
            if conn_details.get("802-11-wireless.mode") == "ap":
                return ConnectionInfo(
                    mode="hotspot",
                    ssid=conn_details.get("802-11-wireless.ssid")
                )
            return ConnectionInfo(
                mode="wifi",
                ssid=conn_details.get("802-11-wireless.ssid")
            )
    except Exception as e:
        logger.warning("Failed to get current connection: %s", e)

    return ConnectionInfo(mode="disconnected", ssid=None)


def get_visible_ssids() -> set[str]:
    """Return a set of SSIDs currently visible in a Wi-Fi scan.

    Uses nmcli.device.wifi() with rescan=True to trigger a fresh scan.
    Falls back to cached results (rescan=False) if the scan request fails.

    Returns:
        Set of SSID strings from the current Wi-Fi scan results.
    """
    try:
        wifi_list = nmcli.device.wifi(rescan=True)
    except Exception:
        try:
            wifi_list = nmcli.device.wifi(rescan=False)
        except Exception as e:
            logger.warning("Failed to get visible SSIDs: %s", e)
            return set()

    return {ap.ssid for ap in wifi_list if ap.ssid}


def connect_wifi_with_fallback(config: AppConfig, current: ConnectionInfo) -> bool:
    """Try to connect to configured Wi-Fi network.

    Iterates through all configured Wi-Fi networks in range until one connects
    successfully. If none are reachable, also tries any Wi-Fi client profiles
    present in NetworkManager but not configured in config file.

    Args:
        config:  AppConfig with list of Wi-Fi network settings.
        current: Current connection info (mode and SSID).

    Returns:
        True if a Wi-Fi connection is active after this call, False otherwise.
    """
    visible_ssids = get_visible_ssids()

    # Build ordered list of SSIDs to try: config networks first, then NM-only profiles
    config_ssids = [
        str(wifi.ssid)
        for wifi in config.network.wifi
        if wifi.ssid and wifi.password
    ]

    nm_only_ssids: list[str] = []
    try:
        for connection in nmcli.connection():
            if connection.conn_type != "wifi":
                continue
            if connection.name in config_ssids:
                continue
            try:
                conn_details = nmcli.connection.show(connection.name)
            except Exception:
                continue
            if conn_details.get("802-11-wireless.mode") != "ap":
                nm_only_ssids.append(connection.name)
    except Exception as e:
        logger.warning("Failed to retrieve NetworkManager profiles: %s", e)

    ssids_to_try = config_ssids + nm_only_ssids

    for ssid_str in ssids_to_try:
        if current["mode"] == "wifi" and current["ssid"] == ssid_str:
            logger.debug("Already connected to '%s'.", ssid_str)
            return True

        if ssid_str not in visible_ssids:
            logger.info("Wi-Fi '%s' not in range, skipping.", ssid_str)
            continue

        try:
            if current["mode"] == "hotspot":
                nmcli.device.disconnect("wlan0")
            nmcli.connection.up(ssid_str)
            logger.info("Successfully connected to Wi-Fi '%s'.", ssid_str)
            return True
        except Exception as e:
            logger.warning("Failed to connect to Wi-Fi '%s': %s. Trying next...", ssid_str, e)
            current = get_current_connection()

    logger.warning("No reachable Wi-Fi network found in config or NetworkManager profiles.")
    return False


def set_up_network(config: AppConfig, activate_network: bool = False) -> None:
    """Create and optionally activate Wi-Fi network connections based on config.

    Creates hotspot and/or Wi-Fi connection profiles in NetworkManager based
    on the active config. If 'activate_network' is True, also switches the
    wlan0 interface to the configured mode.

    Args:
        config:           AppConfig with network settings.
        activate_network: If True, activate the configured network connection.
    """
    if config.network.hotspot.ssid:
        create_hotspot(config)

    if config.network.wifi:
        create_wifi(config)

    if not activate_network:
        return

    current = get_current_connection()

    if config.network.mode == "hotspot":
        hotspot_ssid = config.network.hotspot.ssid
        if not hotspot_ssid:
            return
        if current["mode"] != "hotspot" or current["ssid"] != hotspot_ssid:
            try:
                nmcli.device.disconnect("wlan0")
                nmcli.connection.up(hotspot_ssid)
            except Exception as e:
                logger.warning("Failed to activate hotspot '%s': %s", hotspot_ssid, e)

    elif config.network.mode == "wifi":
        connect_wifi_with_fallback(config, current)


def create_hotspot(config: AppConfig) -> None:
    """Create or update Wi-Fi hotspot connection profile in NetworkManager.

    Deletes any existing hotspot profiles that don't match the configured SSID.
    Creates a new profile if none exists, or updates the password if it changed.

    Args:
        config: AppConfig with hotspot SSID and password settings.
    """
    ssid = config.network.hotspot.ssid
    password = config.network.hotspot.password

    if not ssid or not password:
        return

    ssid_str = str(ssid)
    password_str = str(password)

    # Delete any existing hotspot profiles that don't match the configured hotspot SSID
    for connection in nmcli.connection():
        if connection.conn_type == "wifi":
            conn_name = connection.name
            conn_details = nmcli.connection.show(conn_name)
            if (conn_details.get("802-11-wireless.mode") == "ap"
                and conn_details.get("802-11-wireless.ssid") != ssid_str):
                nmcli.connection.delete(conn_name)

    try:
        conn_details = nmcli.connection.show(ssid_str, show_secrets=True)
    except Exception:
        conn_details = None

    if conn_details is None:
        nmcli.connection.add(
            conn_type="wifi",
            ifname="wlan0",
            name=ssid_str,
            autoconnect=True,
            options={
                "wifi.mode": "ap",
                "wifi.band": "bg",
                "ipv4.method": "shared",
                "ssid": ssid_str,
                "wifi-sec.key-mgmt": "wpa-psk",
                "wifi-sec.psk": password_str,
            }
        )
    elif conn_details.get("802-11-wireless-security.psk") != password_str:
        nmcli.connection.modify(
            name=ssid_str,
            options={
                "wifi.mode": "ap",
                "wifi.band": "bg",
                "ipv4.method": "shared",
                "wifi-sec.key-mgmt": "wpa-psk",
                "wifi-sec.psk": password_str,
                "connection.autoconnect": "yes",
            }
        )


def create_wifi(config: AppConfig) -> None:
    """Create or update Wi-Fi connection profiles in NetworkManager.

    Only syncs from config to NetworkManager (one-way). Existing profiles
    not present in the config are intentionally left untouched to avoid
    removing profiles configured via other means (e.g. RPi Imager).

    Iterates over all configured Wi-Fi networks in order. The first entry
    receives the highest priority (100), each subsequent entry receives 10
    less (90, 80, ...). Skips entries without both SSID and password.

    Args:
        config: AppConfig with list of Wi-Fi network settings.
    """
    for i, wifi in enumerate(config.network.wifi):
        if not wifi.ssid or not wifi.password:
            continue

        ssid_str = str(wifi.ssid)
        password_str = str(wifi.password)
        # Set priority based on index (= position in list)
        priority_str = str(100 - i * 10)

        try:
            conn_details = nmcli.connection.show(ssid_str, show_secrets=True)
        except Exception:
            conn_details = None

        if conn_details is None:
            nmcli.connection.add(
                conn_type="wifi",
                ifname="wlan0",
                name=ssid_str,
                autoconnect=True,
                options={
                    "ssid": ssid_str,
                    "wifi-sec.key-mgmt": "wpa-psk",
                    "wifi-sec.psk": password_str,
                    "connection.autoconnect-priority": priority_str,
                }
            )
        elif (conn_details.get("802-11-wireless-security.psk") != password_str
              or conn_details.get("connection.autoconnect-priority") != priority_str):
            nmcli.connection.modify(
                name=ssid_str,
                options={
                    "wifi-sec.key-mgmt": "wpa-psk",
                    "wifi-sec.psk": password_str,
                    "connection.autoconnect-priority": priority_str,
                    "connection.autoconnect": "yes",
                }
            )
