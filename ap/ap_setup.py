import subprocess
import os

interface = "ap0"
password = "jetcobot"

result = open('/sys/class/net/eth0/address').read().strip()
mac_address = result.strip().replace(":", "")

ssid = f"jetcobot_{mac_address[-4:]}"

save_path = "/home/jetcobot"

hostapd_config = f"""interface={interface}
ssid={ssid}
hw_mode=g
channel=6
ieee80211n=1
wmm_enabled=1
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase={password}
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
"""

with open("/etc/hostapd/hostapd.conf", "w") as f:
    f.write(hostapd_config)

subprocess.run(["sudo", "systemctl", "restart", "hostapd"])

