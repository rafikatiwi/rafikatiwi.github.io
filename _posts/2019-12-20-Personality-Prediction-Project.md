---
layout: post
title: "Personality Prediction Project"
date: 2019-12-20
---

This is a final project of Sensor Data Science course which I took on Fall Semester 2019 at KAIST. By using user's smartphone usage data and sensor data of one week, I tried to predict what kind of personality tendency that user has. I based the personality type with Big Five personality, that consisted of Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. Thus, each user will be predicted for his/her each type of personality's tendency. 

The Datasets used:

AppUsageEventEntity data: This dataset tells about the running applications on user's smartphone and its status (whether it's in a foreground or background) at certain timestamp.
LocationEntity: This dataset tells about user's current location based on their current GPS latitude and longitude at certain timestamp.
DeviceEventEntity: This dataset tells about user's current status of his/her device (e.g: unlocked, screen_off, screen_on, power_disconnected, power_connected, etc) at certain timestamp.
PhysicalActivityEventEntity: This dataset tells about the activity recognition of user at certain timestamps (e.g: still, walking, on_foot, in_vehicle, on_bicycle, running, etc) 
WifiEntity: This dataset tells about the ssid of WiFi connection a user is connected at certain timestamps
DataTrafficEntity: This dataset tells about the volume of receiving and transferring mobile data at certain timestamps (in Kb)
CallLogEntity: This dataset tells about incoming, outgoing, or missed calls a user had at certain timestamps
MessageEntity: This dataset tells about inbox, sent, or outbox text messages a user had at certain timestamps
Distance: This dataset tells about cumulative distance a user reached at certain timestamps
Pedometer: This dataset tells about cumulative step a user reached at certain timestamps
BatteryEntity: This dataset tells about current status of a user's battery (discharged, charging, full) at certain timestamps

The Data Science Process that I did: 
1. Pre-processing


2. Feature Engineering

3. Standard Scaling

4. Predictive Model Building
