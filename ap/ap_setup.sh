#!/bin/bash


sudo /sbin/iw dev wlan0 interface add ap0 type __ap
sleep 1
sudo /sbin/ifconfig ap0 192.168.6.1/24 netmask 255.255.255.0

sudo /sbin/ifconfig ap0 up
sudo /sbin/iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE

sudo /usr/bin/nmcli dev set ap0 managed no

/usr/bin/python3 -u /home/jetcobot/ap/ap_setup.py
