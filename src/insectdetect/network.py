"""Utility functions for network management via NetworkManager.

Source:   https://github.com/maxsitt/insect-detect
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
Docs:     https://maxsitt.github.io/insect-detect-docs/

Functions:
    get_wifi_profiles(): Get all Wi-Fi connection profiles that exist in NetworkManager.
    get_current_connection(): Get the Wi-Fi profile currently active on the wlan0 interface.
    get_ip_address(): Get the IPv4 address assigned to the wlan0 wireless interface.
    enable_wifi_radio(): Enable the Wi-Fi radio if it is disabled in NetworkManager.
    wait_for_wlan0(): Wait until the wlan0 wireless interface is available in NetworkManager.
    get_visible_ssids(): Get the SSIDs currently visible in a Wi-Fi scan and whether the scan completed.
    create_hotspot(): Create or update the Wi-Fi hotspot connection profile in NetworkManager.
    create_wifi(): Create or update Wi-Fi connection profiles in NetworkManager.
    delete_hotspot_profiles(): Delete all hotspot profiles from NetworkManager.
    connect_wifi(): Try to connect to a configured Wi-Fi network.
    activate_hotspot(): Activate the configured RPi Wi-Fi hotspot.
    activate_network_mode(): Switch the wlan0 interface to the network mode set in the config.
    set_up_network(): Set up Wi-Fi connection profiles in NetworkManager and optionally activate the mode.
"""

import logging
import socket
import time
from dataclasses import dataclass

import nmcli
import psutil

from insectdetect.config import AppConfig
from insectdetect.constants import WIFI_CONNECT_TIMEOUT, WIFI_SCAN_ATTEMPTS, WLAN0_READY_TIMEOUT

# Initialize logger for this module
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WifiProfile:
    """Wi-Fi connection profile stored in NetworkManager."""
    uuid: str
    name: str
    ssid: str | None
    is_hotspot: bool


def get_wifi_profiles() -> list[WifiProfile]:
    """Get all Wi-Fi connection profiles that exist in NetworkManager.

    Returns:
        List of WifiProfile entries in the order reported by NetworkManager. Profiles whose
        details could not be retrieved are skipped, the list is empty if NetworkManager
        could not be queried at all.
    """
    profiles: list[WifiProfile] = []
    try:
        connections = nmcli.connection()
    except Exception:
        logger.exception("Failed to retrieve NetworkManager profiles")
        return profiles

    for connection in connections:
        if connection.conn_type != "wifi":
            continue
        try:
            conn_details = nmcli.connection.show(connection.uuid)
        except Exception:
            logger.exception("Failed to get details of profile '%s', skipping it", connection.name)
            continue
        profiles.append(WifiProfile(
            uuid=connection.uuid,
            name=connection.name,
            ssid=conn_details.get("802-11-wireless.ssid"),
            is_hotspot=conn_details.get("802-11-wireless.mode") == "ap"
        ))

    return profiles


def get_current_connection(profiles: list[WifiProfile] | None = None) -> WifiProfile | None:
    """Get the Wi-Fi profile currently active on the wlan0 interface.

    Args:
        profiles: Wi-Fi profiles in NetworkManager, used to look up the active connection.

    Returns:
        Currently active WifiProfile, or None if wlan0 is disconnected or could not be queried.
    """
    try:
        active_connections = nmcli.connection.show_all(active=True)
    except Exception:
        logger.exception("Failed to get currently active connection")
        return None

    connection = next((c for c in active_connections if c.device == "wlan0"), None)
    if connection is None:
        return None

    if profiles is not None:
        profile = next((p for p in profiles if p.uuid == connection.uuid), None)
        if profile is not None:
            return profile

    try:
        conn_details = nmcli.connection.show(connection.uuid)
    except Exception:
        logger.exception("Failed to get details of active profile '%s'", connection.name)
        return None

    return WifiProfile(
        uuid=connection.uuid,
        name=connection.name,
        ssid=conn_details.get("802-11-wireless.ssid"),
        is_hotspot=conn_details.get("802-11-wireless.mode") == "ap"
    )


def get_ip_address() -> str:
    """Get the IPv4 address assigned to the wlan0 wireless interface.

    Read from the kernel, which also works in hotspot mode and without NetworkManager.

    Returns:
        IPv4 address string, or '127.0.0.1' if wlan0 has no IPv4 address assigned.
    """
    try:
        for address in psutil.net_if_addrs().get("wlan0", []):
            if address.family == socket.AF_INET:
                return address.address
    except Exception:
        logger.exception("Failed to get IP address of wlan0 interface")

    return "127.0.0.1"


def enable_wifi_radio() -> bool:
    """Enable the Wi-Fi radio if it is disabled in NetworkManager.

    A disabled radio keeps wlan0 unavailable, which makes every scan and connection fail.

    Returns:
        True if the Wi-Fi radio was already enabled or the command to enable it succeeded,
        False if it could not be checked or enabled.
    """
    try:
        if nmcli.radio.wifi():
            return True

        logger.info("Wi-Fi radio is disabled, enabling it")
        nmcli.radio.wifi_on()
        return True
    except Exception:
        logger.exception("Failed to check/enable Wi-Fi radio")
        return False


def wait_for_wlan0(timeout: int = WLAN0_READY_TIMEOUT) -> bool:
    """Wait until the wlan0 wireless interface is available in NetworkManager.

    Enables the Wi-Fi radio first if it is disabled. At startup the interface can still be
    unavailable for several seconds, which would make all connection attempts fail immediately.

    Args:
        timeout: Maximum time (seconds) to wait for the interface to become available.

    Returns:
        True if wlan0 is available, False if it is still not available after 'timeout'.
    """
    enable_wifi_radio()

    start = time.monotonic()
    deadline = start + timeout
    while True:
        try:
            device_wlan0 = next((dev for dev in nmcli.device() if dev.device == "wlan0"), None)
            if device_wlan0 and device_wlan0.state not in ("unavailable", "unmanaged"):
                waited = time.monotonic() - start
                if waited >= 1:
                    logger.info("wlan0 interface became available after %.1f seconds", waited)
                return True
        except Exception as e:
            logger.debug("Failed to get status of wlan0 interface: %s", e)

        if time.monotonic() >= deadline:
            logger.warning("wlan0 interface not available after %s seconds", timeout)
            return False
        time.sleep(1)


def get_visible_ssids(attempts: int = WIFI_SCAN_ATTEMPTS) -> tuple[set[str], bool]:
    """Get the SSIDs currently visible in a Wi-Fi scan and whether the scan completed.

    A scan that finds networks is conclusive and returns immediately. An empty result is
    only conclusive after two completed scans, as a single scan can miss networks that are
    in range. A scan that was refused or failed does not count as completed and is retried.

    Args:
        attempts: Maximum number of scan attempts before giving up.

    Returns:
        Tuple of (visible SSIDs, True if the scan produced a conclusive result: either
        networks were found, or two completed scans found none). An empty set with False
        means that the scan did not work, not that no Wi-Fi network is in range.
    """
    completed_scans = 0
    for attempt in range(1, attempts + 1):
        try:
            wifi_list = nmcli.device.wifi(ifname="wlan0", rescan=True)
            visible_ssids = {ap.ssid for ap in wifi_list if ap.ssid}
        except Exception as e:
            logger.debug("Wi-Fi scan failed (attempt %s/%s): %s", attempt, attempts, e)
        else:
            if visible_ssids:
                return visible_ssids, True

            completed_scans += 1
            if completed_scans >= 2:
                logger.info("No Wi-Fi network in range (%s completed scans)", completed_scans)
                return set(), True

    logger.warning("Wi-Fi scan did not work after %s attempts", attempts)
    return set(), False


def create_hotspot(config: AppConfig, profiles: list[WifiProfile] | None = None) -> bool:
    """Create or update the Wi-Fi hotspot connection profile in NetworkManager.

    An existing profile is matched by its SSID and updated in place, hotspot profiles
    with a different SSID are deleted.

    Autoconnect is disabled, as NetworkManager considers a hotspot always available while
    a Wi-Fi network is only available once it was found in a scan. A hotspot with enabled
    autoconnect can therefore win the race against Wi-Fi networks that are still being
    scanned for. It is activated by activate_hotspot() instead.

    Args:
        config:   AppConfig with hotspot SSID and password settings.
        profiles: Wi-Fi profiles that exist in NetworkManager, retrieved if not provided.

    Returns:
        True if the profile was created or modified, False if it was already up to date
        or could not be checked.

    Raises:
        Exception: If the profile could not be created or updated in NetworkManager.
    """
    if profiles is None:
        profiles = get_wifi_profiles()

    ssid_str = config.network.hotspot.effective_ssid
    password_str = config.network.hotspot.effective_password
    # Set lowest possible priority to never outrank a Wi-Fi network if autoconnect is re-enabled
    priority_str = "-999"

    hotspot_profile = None
    for profile in profiles:
        if not profile.is_hotspot:
            continue
        if profile.ssid == ssid_str:
            hotspot_profile = profile
            continue
        # Delete any existing hotspot profiles that don't match the configured hotspot SSID
        try:
            logger.info("Deleting outdated hotspot profile '%s'", profile.name)
            nmcli.connection.delete(profile.uuid)
        except Exception:
            logger.exception("Failed to delete profile '%s'", profile.name)

    if hotspot_profile is None:
        nmcli.connection.add(
            conn_type="wifi",
            ifname="wlan0",
            name=ssid_str,
            autoconnect=False,
            options={
                "wifi.mode": "ap",
                "wifi.band": "bg",
                "ipv4.method": "shared",
                "ssid": ssid_str,
                "wifi-sec.key-mgmt": "wpa-psk",
                "wifi-sec.psk": password_str,
                "connection.autoconnect-priority": priority_str,
            }
        )
        return True

    try:
        conn_details = nmcli.connection.show(hotspot_profile.uuid, show_secrets=True)
    except Exception:
        logger.exception("Failed to get details of hotspot profile '%s'", ssid_str)
        return False

    if (conn_details.get("802-11-wireless-security.psk") != password_str
        or conn_details.get("connection.autoconnect") != "no"
        or conn_details.get("connection.autoconnect-priority") != priority_str):
        nmcli.connection.modify(
            name=hotspot_profile.uuid,
            options={
                "wifi.mode": "ap",
                "wifi.band": "bg",
                "ipv4.method": "shared",
                "wifi-sec.key-mgmt": "wpa-psk",
                "wifi-sec.psk": password_str,
                "connection.autoconnect": "no",
                "connection.autoconnect-priority": priority_str,
            }
        )
        return True

    return False


def create_wifi(config: AppConfig, profiles: list[WifiProfile] | None = None) -> None:
    """Create or update Wi-Fi connection profiles in NetworkManager.

    Only syncs from config to NetworkManager (one-way): profiles that are not in the config
    are left untouched, to keep networks set up via other means (e.g. RPi Imager). Existing
    profiles are matched by SSID and updated in place, never duplicated.

    Networks are prioritized by their position in the config list, the first entry receives
    the highest priority (100) and each subsequent entry 10 less (90, 80, ...).

    Args:
        config:   AppConfig with list of Wi-Fi network settings.
        profiles: Wi-Fi profiles that exist in NetworkManager, retrieved if not provided.
    """
    if profiles is None:
        profiles = get_wifi_profiles()

    wifi_profiles = [p for p in profiles if not p.is_hotspot]

    for i, wifi in enumerate(config.network.wifi):
        if not wifi.ssid or not wifi.password:
            continue

        ssid_str = str(wifi.ssid)
        password_str = str(wifi.password)
        priority_str = str(100 - i * 10)

        wifi_profile = next((p for p in wifi_profiles if p.ssid == ssid_str), None)

        if wifi_profile is None:
            try:
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
            except Exception:
                logger.exception("Failed to create Wi-Fi profile '%s'", ssid_str)
            continue

        try:
            conn_details = nmcli.connection.show(wifi_profile.uuid, show_secrets=True)
        except Exception:
            logger.exception("Failed to get details of Wi-Fi profile '%s'", ssid_str)
            continue

        if (conn_details.get("802-11-wireless-security.psk") != password_str
            or conn_details.get("connection.autoconnect-priority") != priority_str
            or conn_details.get("connection.autoconnect") != "yes"):
            try:
                nmcli.connection.modify(
                    name=wifi_profile.uuid,
                    options={
                        "wifi-sec.key-mgmt": "wpa-psk",
                        "wifi-sec.psk": password_str,
                        "connection.autoconnect-priority": priority_str,
                        "connection.autoconnect": "yes",
                    }
                )
            except Exception:
                logger.exception("Failed to update Wi-Fi profile '%s'", ssid_str)


def delete_hotspot_profiles(profiles: list[WifiProfile] | None = None) -> None:
    """Delete all hotspot profiles from NetworkManager.

    Called if the hotspot is neither set as network mode nor enabled as fallback.
    A leftover profile would either never be activated or, if it has autoconnect
    enabled, could take precedence over Wi-Fi networks at boot.

    Args:
        profiles: Wi-Fi profiles that exist in NetworkManager, retrieved if not provided.
    """
    if profiles is None:
        profiles = get_wifi_profiles()

    for profile in profiles:
        if not profile.is_hotspot:
            continue
        try:
            logger.info("Deleting hotspot profile '%s', as no hotspot is configured",
                        profile.name)
            nmcli.connection.delete(profile.uuid)
        except Exception:
            logger.exception("Failed to delete hotspot profile '%s'", profile.name)


def _get_client_profiles(
    config: AppConfig,
    profiles: list[WifiProfile]
) -> tuple[list[WifiProfile], list[WifiProfile]]:
    """Collect all Wi-Fi client profiles that can be used to connect, ordered by priority.

    Hotspot profiles are skipped, as they are only activated by activate_hotspot().
    Configured Wi-Fi networks are returned in the order of the config file,
    all remaining profiles in the order reported by NetworkManager.

    Args:
        config:   AppConfig with list of Wi-Fi network settings.
        profiles: Wi-Fi profiles that exist in NetworkManager.

    Returns:
        Tuple of (profiles configured in config file, profiles only present in NetworkManager).
    """
    wifi_profiles = [p for p in profiles if not p.is_hotspot and p.ssid]

    config_profiles: list[WifiProfile] = []
    for wifi in config.network.wifi:
        if not wifi.ssid or not wifi.password:
            continue
        matching = [p for p in wifi_profiles if p.ssid == str(wifi.ssid)]
        if matching:
            config_profiles += matching
        else:
            logger.warning("No NetworkManager profile found for configured Wi-Fi '%s'", wifi.ssid)

    config_uuids = {p.uuid for p in config_profiles}
    nm_only_profiles = [p for p in wifi_profiles if p.uuid not in config_uuids]

    return config_profiles, nm_only_profiles


def connect_wifi(
    config: AppConfig,
    current: WifiProfile | None,
    profiles: list[WifiProfile] | None = None
) -> bool:
    """Try to connect to a configured Wi-Fi network.

    If the interface is already connected to a known network (from the config file or
    NetworkManager), that connection is kept, even if a network with a higher priority is
    in range, as switching would drop a working connection.

    Otherwise all configured Wi-Fi networks that are in range are tried in the order of
    the config file, followed by any client profiles that only exist in NetworkManager.
    If the scan itself did not work, the configured networks are still tried, as an
    explicit connection request makes NetworkManager scan for the network itself.

    Args:
        config:   AppConfig with list of Wi-Fi network settings.
        current:  Wi-Fi profile that is currently active, or None if disconnected.
        profiles: Wi-Fi profiles that exist in NetworkManager, retrieved if not provided.

    Returns:
        True if a Wi-Fi connection is active after this call, False otherwise.
    """
    if profiles is None:
        profiles = get_wifi_profiles()

    config_profiles, nm_only_profiles = _get_client_profiles(config, profiles)
    if not config_profiles and not nm_only_profiles:
        logger.warning("No Wi-Fi network configured in config file or NetworkManager profiles")
        return False

    known_ssids = {p.ssid for p in config_profiles + nm_only_profiles}
    if current and not current.is_hotspot and current.ssid in known_ssids:
        logger.info("Already connected to Wi-Fi '%s'", current.ssid)
        return True

    visible_ssids, scan_completed = get_visible_ssids()
    if visible_ssids:
        profiles_to_try = []
        for profile in config_profiles + nm_only_profiles:
            if profile.ssid in visible_ssids:
                profiles_to_try.append(profile)
            else:
                logger.info("Wi-Fi '%s' not in range, skipping", profile.ssid)
    elif scan_completed:
        # No Wi-Fi network is in range, connection attempts would only run into their timeout
        logger.info("No configured Wi-Fi network is in range")
        profiles_to_try = []
    else:
        # Scan did not work, still try the explicitly configured Wi-Fi networks
        logger.warning("Trying all configured Wi-Fi networks despite failed Wi-Fi scan")
        profiles_to_try = config_profiles or nm_only_profiles

    if not profiles_to_try:
        logger.warning("No reachable Wi-Fi network found in config or NetworkManager profiles")
        return False

    for profile in profiles_to_try:
        try:
            nmcli.connection.up(profile.uuid, wait=WIFI_CONNECT_TIMEOUT)
            logger.info("Successfully connected to Wi-Fi '%s'", profile.ssid)
            return True
        except Exception as e:
            logger.warning("Failed to connect to Wi-Fi '%s': %s. Trying next...", profile.ssid, e)

    # NetworkManager could have autoconnected to a Wi-Fi network in the meantime
    current = get_current_connection(profiles)
    if current and not current.is_hotspot:
        logger.info("Connected to Wi-Fi '%s' by NetworkManager", current.ssid)
        return True

    logger.warning("Could not connect to any Wi-Fi network that is in range")
    return False


def activate_hotspot(
    config: AppConfig,
    reactivate: bool = False,
    profiles: list[WifiProfile] | None = None
) -> bool:
    """Activate the configured RPi Wi-Fi hotspot.

    The hotspot is never activated automatically by NetworkManager (see create_hotspot()),
    so this is the only way it is started. NetworkManager applies changed settings only
    when a profile is activated, which is what 'reactivate' is for - an already active
    hotspot would otherwise keep the old password until the next reboot.

    Args:
        config:     AppConfig with hotspot settings.
        reactivate: If True, restart the hotspot if it is already active.
        profiles:   Wi-Fi profiles that exist in NetworkManager, retrieved if not provided.

    Returns:
        True if the hotspot is active after this call, False otherwise.
    """
    if profiles is None:
        profiles = get_wifi_profiles()

    ssid_str = config.network.hotspot.effective_ssid
    profile = next((p for p in profiles if p.is_hotspot and p.ssid == ssid_str), None)
    if profile is None:
        logger.warning("No hotspot profile found for SSID '%s', cannot activate it", ssid_str)
        return False

    current = get_current_connection(profiles)
    if current and current.is_hotspot and current.ssid == ssid_str:
        if not reactivate:
            logger.info("Hotspot '%s' is already active", ssid_str)
            return True
        logger.info("Reactivating hotspot '%s' to apply changed settings", ssid_str)
        try:
            nmcli.connection.down(profile.uuid, wait=WIFI_CONNECT_TIMEOUT)
        except Exception:
            logger.exception("Failed to deactivate hotspot '%s'", ssid_str)

    try:
        nmcli.connection.up(profile.uuid, wait=WIFI_CONNECT_TIMEOUT)
        logger.info("Successfully activated hotspot '%s'", ssid_str)
        return True
    except Exception:
        logger.exception("Failed to activate hotspot '%s'", ssid_str)
        return False


def activate_network_mode(
    config: AppConfig,
    hotspot_changed: bool = False,
    profiles: list[WifiProfile] | None = None
) -> None:
    """Switch the wlan0 interface to the network mode set in the config.

    In 'wifi' mode, the hotspot is activated as fallback if no Wi-Fi network can be
    reached and 'network.hotspot.fallback' is enabled.

    Args:
        config:          AppConfig with network settings.
        hotspot_changed: If True, restart an already active hotspot to apply its new settings.
        profiles:        Wi-Fi profiles that exist in NetworkManager, retrieved if not provided.
    """
    wait_for_wlan0()

    if profiles is None:
        profiles = get_wifi_profiles()

    if config.network.mode == "hotspot":
        activate_hotspot(config, reactivate=hotspot_changed, profiles=profiles)
        return

    if not connect_wifi(config, get_current_connection(profiles), profiles):
        if not config.network.hotspot.fallback:
            logger.warning("Hotspot fallback is disabled, device is not accessible via Wi-Fi")
            return
        logger.info("Activating RPi hotspot as fallback to keep the device accessible")
        activate_hotspot(config, reactivate=hotspot_changed, profiles=profiles)


def set_up_network(config: AppConfig, activate_network: bool = False) -> None:
    """Set up Wi-Fi connection profiles in NetworkManager and optionally activate the mode.

    The hotspot profile is created/updated if the hotspot is set as network mode or
    enabled as fallback, otherwise it is deleted. Wi-Fi profiles are only created/updated
    if 'startup.wifi_sync' is enabled. If 'activate_network' is True, the wlan0 interface
    is switched to the configured mode, even if creating or updating the profiles failed.

    Args:
        config:           AppConfig with network settings.
        activate_network: If True, activate the configured network connection.
    """
    profiles = get_wifi_profiles()

    hotspot_changed = False
    try:
        if config.network.hotspot_enabled:
            hotspot_changed = create_hotspot(config, profiles)
        else:
            delete_hotspot_profiles(profiles)
    except Exception:
        logger.exception("Failed to create/update RPi hotspot profile in NetworkManager")

    if config.startup.wifi_sync.enabled and config.network.wifi:
        try:
            create_wifi(config, profiles)
        except Exception:
            logger.exception("Failed to create/update Wi-Fi profiles in NetworkManager")

    if activate_network:
        # Retrieve profiles again, as profiles could have been added or deleted above
        activate_network_mode(config, hotspot_changed, get_wifi_profiles())
