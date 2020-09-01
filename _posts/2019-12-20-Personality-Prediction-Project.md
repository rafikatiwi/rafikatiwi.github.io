---
layout: post
title: "Personality Prediction Project"
date: 2019-12-20
---

This is a final project of Sensor Data Science course which I took on Fall Semester 2019 at KAIST. By using user's smartphone usage data and sensor data of one week, I tried to predict what kind of personality tendency that user has. I based the personality type with Big Five personality, that consisted of Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. Thus, each user will be predicted for his/her each type of personality's tendency. 

The Datasets used:

AppUsageEventEntity data: This dataset tells about the running applications on user's smartphone and its status (whether it's in a foreground or background) at certain timestamp. <br/>
LocationEntity: This dataset tells about user's current location based on their current GPS latitude and longitude at certain timestamp. <br/>
DeviceEventEntity: This dataset tells about user's current status of his/her device (e.g: unlocked, screen_off, screen_on, power_disconnected, power_connected, etc) at certain timestamp. <br/>
PhysicalActivityEventEntity: This dataset tells about the activity recognition of user at certain timestamps (e.g: still, walking, on_foot, in_vehicle, on_bicycle, running, etc) <br/>
WifiEntity: This dataset tells about the ssid of WiFi connection a user is connected at certain timestamps <br/>
DataTrafficEntity: This dataset tells about the volume of receiving and transferring mobile data at certain timestamps (in Kb) <br/>
CallLogEntity: This dataset tells about incoming, outgoing, or missed calls a user had at certain timestamps <br/>
MessageEntity: This dataset tells about inbox, sent, or outbox text messages a user had at certain timestamps <br/>
Distance: This dataset tells about cumulative distance a user reached at certain timestamps <br/>
Pedometer: This dataset tells about cumulative step a user reached at certain timestamps <br/>
BatteryEntity: This dataset tells about current status of a user's battery (discharged, charging, full) at certain timestamps <br/>

The Data Science Process that I did: 
1. Pre-processing (Feature Extraction)

I did feature extractions for each of the dataset. There are two phases of feature extractions. First phase is to extract features from the original datasets and the second phase is to extract features by applying time windowing for each datasets and extract the statistics value such as (mean, median, max, min, standard deviations, etc) as features. 

To simplify what kind of Apps each user is using, I identified the category of the Apps by calling google_play_scraper python library. The google_play_scraper will take application package as input and return the category of the application. I extracted the category of the App to understand what type of Apps a user usually uses. 

```
#fetch the app category name
application_packages_list = application_packages.tolist()
application_packages_dict = dict((el,"") for el in application_packages_list)
for application_package in application_packages_list:
    try:
        application_packages_dict[application_package] = play_scraper.details(application_package)['category'][0]
    except: 
        pass #leave the category empty string if the category of the app can't be recognized
```
```
#get category of the app and save it as new csv files 

for participant_ID in participant_IDs:
    ID_temp = pd.read_csv("Sensor Data Used/" + participant_ID + "/AppUsageEventEntity.csv")
    ID_temp['AppCategory'] = ""
    ID_temp['AppCategory'] = ID_temp.apply(lambda row: application_packages_dict[row['packageName']], axis=1)
    ID_temp.to_csv(r'Sensor Data Used/' + participant_ID + '/AppUsageEventEntity_withCat.csv')
    
    ID_stat_temp = pd.read_csv("Sensor Data Used/" + participant_ID + "/AppUsageStatEntity.csv")
    ID_stat_temp['AppCategory'] = ""
    ID_stat_temp['AppCategory'] = ID_stat_temp.apply(lambda row: application_packages_dict[row['packageName']], axis=1)
    ID_stat_temp.to_csv(r'Sensor Data Used/' + participant_ID + '/AppUsageStatEntity_withCat.csv')
```

Next, I extracted the category of a location a user visited at certain timestamp using OpenStreetMap
```
participant_IDs = []
geopy.geocoders.options.default_timeout = 1
geolocator = Nominatim(user_agent="personality_detection")
for participant_ID in participant_IDs:
    ID_temp = pd.read_csv("Sensor Data Used/" + participant_ID + "/LocationEntity.csv")
    ID_temp['accuracy'] = pd.to_numeric(ID_temp['accuracy'],errors='coerce')
    ID_temp['PlaceType'] = ""
    added = 0
    failed = 0
    for index, row in ID_temp.iterrows():
        if row['accuracy'] < 15:
            try:
                location_temp = geolocator.reverse(row['latitude'] + "," + row['longitude'])
                ID_temp.loc[index, 'PlaceType'] = list(location_temp.raw['address'].keys())[0:1][0]
                print(index)
                added +=1
                print(participant_ID + ' ' + str(index) + ' added') 
            except:
                ID_temp.loc[index, 'PlaceType'] = ""
                failed += 1
                pass
    ID_temp.to_csv(r'Sensor Data Used/' + participant_ID + '/LocationEntity_withPlaceType.csv')
```
On the second phase of feature extraction, for each datasets, I did time windowing feature extractions of one day (24 hours) and sliding window of 0.5 and extract the statistics values. <br/>

Extract time window features from AppUsageEventEntity data

```
#Take apps whose type is MOVE_TO_FOREGROUND or USER_INTERACTION
#Categorize 'GAME_ROLE_PLAYING', 'GAME_ACTION', 'GAME_CASUAL', 'GAME_CARD', 'GAME_SIMULATION', 'GAME_STRATEGY', 'GAME_RACING', 'GAME_PUZZLE', 'GAME_MUSIC', 'GAME_ADVENTURE', 'GAME_SPORTS', 'GAME_ARCADE', 'GAME_EDUCATIONAL', 'GAME_BOARD' to 'GAME'

def AppUsageEvent_extraction(AppUsageEventEntityData):
    AppUsageEventEntityData.rename(columns={'name':'name_AppUsageEvent'}, inplace=True)
    AppUsageEventEntityData['AppCategory'].fillna("", inplace=True)
    
    for index, row in AppUsageEventEntityData.iterrows():
        if row['AppCategory'] == "":
            AppUsageEventEntityData.loc[index, 'AppCategory'] = row['name_AppUsageEvent']

    ##make subset of the AppUsageEvent data
    AppUsageEventEntityData = AppUsageEventEntityData[AppUsageEventEntityData['AppCategory'].isin(['GAME_EDUCATIONAL',
     'PHOTOGRAPHY', 'TRAVEL_AND_LOCAL', 'BEAUTY', 'GAME_CASUAL',
     'ENTERTAINMENT','GAME_CARD','ART_AND_DESIGN','BOOKS_AND_REFERENCE','GAME_RACING','MAPS_AND_NAVIGATION','MEDICAL',
     'PRODUCTIVITY','FOOD_AND_DRINK','BUSINESS','SPORTS','GAME_SIMULATION','HOUSE_AND_HOME','SHOPPING','EDUCATION',
     'COMICS','LIBRARIES_AND_DEMO','DATING','GAME_MUSIC','TOOLS','GAME_SPORTS','AUTO_AND_VEHICLES',
     'FINANCE','GAME_PUZZLE','WEATHER','GAME_ROLE_PLAYING','HEALTH_AND_FITNESS','GAME_ADVENTURE',
     'LIFESTYLE','GAME_ACTION','COMMUNICATION','GAME_BOARD','MUSIC_AND_AUDIO','GAME_STRATEGY',
     'NEWS_AND_MAGAZINES','SOCIAL','GAME_ARCADE',
     'VIDEO_PLAYERS','PERSONALIZATION', '메시지', '연락처', '인터넷', '전화', 'Samsung Pay', '갤러리', '시계', '캘린더', '리마인더', '파인더',
                                                                                                  '카메라']) ]
    AppUsageEventEntityData = AppUsageEventEntityData[AppUsageEventEntityData['type'].isin(['MOVE_TO_FOREGROUND', 'USER_INTERACTION'])]
    
    ##categroize all game category apps to only one category 'GAME'
    for index, row in AppUsageEventEntityData.iterrows():
        if row['AppCategory'] == "":
            AppUsageEventEntityData.loc[index, 'AppCategory'] = row['name_AppUsageStat']
        elif "GAME" in row['AppCategory']:
            AppUsageEventEntityData.loc[index, 'AppCategory'] = 'GAME'
    
    extracted_AppUsageEventEntityData = extract_features_categorical_data(AppUsageEventEntityData.copy(), 'AppCategory', 24*60, 0.1)
    extracted_AppUsageEventEntityData.drop(['Type'], axis=1, inplace=True)
    extracted_AppUsageEventEntityData = resample_sum(extracted_AppUsageEventEntityData.copy())
    extracted_AppUsageEventEntityData = get_more_features(extracted_AppUsageEventEntityData.copy())
    
    return extracted_AppUsageEventEntityData
```



For this project, I built a predictive model from a time-series data. Thus, I had to set a time window of one day (24 hours) and sliding window of 0.5 for each feature in each of the datasets. 

2. Predictive Model Building

3. Feature Engineering
