---
layout: post
title: "Personality Prediction Project"
date: 2019-12-20
---

This is a final project of Sensor Data Science course which I took on Fall Semester 2019 at KAIST. By using user's smartphone usage data and sensor data, I tried to predict what kind of personality tendency that user has. I based the personality type with Big Five personality, that consisted of Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. Thus, each user will be predicted for his/her each type of personality's tendency. 

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

Extract features using time window 

```
import numpy as np
from typing import Union, Tuple

def duration_subset(from_times: np.ndarray, 
                    to_times: np.ndarray, 
                    values: np.ndarray, 
                    from_boundary: Union[int, float],
                    to_boundary: Union[int, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Handling NaN
    f = np.where(np.isnan(from_times), 0, from_times)
    t = np.where(np.isnan(to_times), np.inf, to_times)
    
    # where -- Return elements chosen from x or y depending on condition.
    # a = np.arange(10) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # np.where(a < 5, a, 10*a)  # array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])
    
    inner = (from_boundary <= f) & (to_boundary >= t)
    outer = (from_boundary > f) & (to_boundary < t)
    left_overlap = (from_boundary < t) & (to_boundary > t) & (from_boundary > f)
    right_overlap = (to_boundary > f) & (from_boundary < f) & (to_boundary < t)

    # Concatenate four conditions
    cond = inner | outer | left_overlap | right_overlap

    f, t, v = f[cond], t[cond], values[cond]
   
    # Change values beyond boundaries into boundary values. 
    # > Clip (limit) the values in an array
    f = np.clip(f, a_min=from_boundary, a_max=None)
    t = np.clip(t, a_min=None, a_max=to_boundary)

    # Find points only that duration is positive.
    non_zero_duration = t - f > 0
    
    return f[non_zero_duration], t[non_zero_duration], v[non_zero_duration]
```

```
def extract_features_categorical_data(dataset_input, feature_input, window_size, overlap_ratio):
    dataset = dataset_input
    feature = feature_input
    diff = dataset.loc[lambda x: x[feature] != x.shift(1)[feature], :] 
    DURATION = pd.concat(
    [diff, diff.shift(-1).rename( lambda x: '_{}'.format(x), axis=1 ) # axis=1 is equivalent to columns=mapper)
], axis=1).assign( 
    start=lambda x: x['timestamp'],  # adding new columns instead of re-renaming 
    end=lambda x: x['_timestamp']
).loc[:, ['start', 'end', feature]] # extracting three columns 

    
    WIN_SIZE_IN_MIN = window_size #@param {type:"slider", min:3, max:10, step:1}
    OVERLAP_RATIO = overlap_ratio #@param {type:"slider", min:0.2, max:0.8, step:0.1}

    WIN_SIZE_IN_MS = WIN_SIZE_IN_MIN * 60 * 1000
    
    DURATION.start = DURATION.start.apply(pd.to_numeric)
    DURATION.end = DURATION.end.apply(pd.to_numeric)
    
    START_TIME, END_TIME = DURATION.loc[:, 'start'].min(), DURATION.loc[:, 'end'].max()

    
    
    WINDOWS = np.arange(START_TIME + WIN_SIZE_IN_MS, END_TIME, WIN_SIZE_IN_MS * (1 - OVERLAP_RATIO))
    EXTRACTED_FEATURES = []

    for w in WINDOWS:
        win_start, win_end = w - WIN_SIZE_IN_MS, w

        start = DURATION.loc[:, 'start'].values
        end = DURATION.loc[:, 'end'].values
        status = DURATION.loc[:, feature].values

        start_bound, end_bound, status_bound = duration_subset(start, end, status, win_start, win_end)

      # If a subset of data is empty, go to the next window.
        if start_bound.shape[0] == 0:
            continue

      # Here, -1 means last element of arrays.
        current_state = status_bound[-1]
        EXTRACTED_FEATURES.append((w, '{}'.format('Type'), current_state))

      # For each status (i.e., SCREEN_ON and SCREEN_OFF), extract duration and frequency.
        for s in DURATION.loc[:, feature].unique():
            cond = status_bound == s
            duration = np.sum(end_bound[cond] - start_bound[cond])
            frequency = status_bound[cond].shape[0]

            EXTRACTED_FEATURES.append((win_end, '{}-{}'.format('Duration', s), duration))
            EXTRACTED_FEATURES.append((win_end, '{}-{}'.format('Frequency', s), frequency))

    EXTRACTED_FEATURES = pd.DataFrame(EXTRACTED_FEATURES, columns=['timestamp', 'feature', 'value'])
    EXTRACTED_FEATURES = EXTRACTED_FEATURES.pivot(index='timestamp', columns='feature', values='value').reset_index()
    return EXTRACTED_FEATURES
```

```
def resample_sum(dataset):
    dataset['timestamp'] = pd.to_datetime(pd.to_numeric(dataset['timestamp']), unit='ms')
    dataset.set_index('timestamp', drop=True, inplace=True)
    resampled_dataset = dataset.resample('D').sum()
    return(resampled_dataset)
```

```
from scipy.stats import skew
def get_more_features(dataset):
    variables = dataset.columns
    new_dataset = []
    for var in variables:
    # select the rows that belong to the current window, w
        value = dataset[var]
          
    # extract basic features 
        min_v = np.min(value) # min
        max_v = np.max(value) # max
        mean_v = np.mean(value) # mean
        std_v = np.std(value) # std. dev.
        median = np.median(value)
        skewness = skew(value)
    
    # append each result (w: current window's end-timestamp, extracted feature) as a new row
        new_dataset.append(('{}-{}'.format('Min', var), min_v))
        new_dataset.append(('{}-{}'.format('Max', var), max_v))
        new_dataset.append(('{}-{}'.format('Mean', var), mean_v))
        new_dataset.append(('{}-{}'.format('Std', var), std_v))
        new_dataset.append(('{}-{}'.format('Median', var), median))
        new_dataset.append(('{}-{}'.format('Skewness', var), skewness))
        
# Reshape data to produce a pivot table based on column values
    new_dataset = pd.DataFrame({x[0]:x[1:] for x in new_dataset})
    return new_dataset
```


Extract time window features from AppUsageEventEntity data

```
#Take apps in which the type is MOVE_TO_FOREGROUND or USER_INTERACTION
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
    
    extracted_AppUsageEventEntityData = extract_features_categorical_data(AppUsageEventEntityData.copy(), 'AppCategory', 24*60, 0.5)
    extracted_AppUsageEventEntityData.drop(['Type'], axis=1, inplace=True)
    extracted_AppUsageEventEntityData = resample_sum(extracted_AppUsageEventEntityData.copy())
    extracted_AppUsageEventEntityData = get_more_features(extracted_AppUsageEventEntityData.copy())
    
    return extracted_AppUsageEventEntityData
```

Extract time window features from LocationEntity data

```
def LocationEntity_extraction(LocationEntity_withPlaceTypeData):
    LocationEntity_withPlaceTypeData['PlaceType'].fillna("", inplace=True)
    
    #simplify the place type
    for index, row in LocationEntity_withPlaceTypeData.iterrows():
        if row['PlaceType'] == 'public_building':
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'building'
        elif (row['PlaceType'] == 'house_number') or (row['PlaceType'] == 'neighbourhood') or (row['PlaceType']== 'residential') or (row['PlaceType']== 'chalet') or (row['PlaceType']== 'castle') or (row['PlaceType']== 'hamlet'):
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'house'
        elif row['PlaceType'] == 'fast_food':
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'restaurant'
        elif row['PlaceType'] == 'bay':
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'beach'
        elif (row['PlaceType'] == 'bicycle_parking') or (row['PlaceType'] == 'bicycle') or (row['PlaceType'] == 'car') or (row['PlaceType'] == 'car_rental') or (row['PlaceType'] == 'car_repair') or (row['PlaceType'] == 'car_wash') or (row['PlaceType'] == 'bus_station') or (row['PlaceType'] == 'bus_stop') or (row['PlaceType'] == 'ferry_terminal') or (row['PlaceType'] == 'subway_entrance') or (row['PlaceType'] == 'station'):
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'transportation'
        elif row['PlaceType'] == 'beverages':
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'cafe'
        elif row['PlaceType'] == 'bus_station':
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'bus_stop'
        elif (row['PlaceType'] == 'butcher') or (row['PlaceType'] == 'convenience') or (row['PlaceType'] == 'supermarket'):
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'grocery'
        elif row['PlaceType'] == 'college':
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'university'
        elif row['PlaceType'] == 'kindergarten':
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'school'
        elif (row['PlaceType'] == 'memorial') or (row['PlaceType'] == 'monument'):
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'museum'
        elif (row['PlaceType'] == 'optician') or (row['PlaceType'] == 'pharmacy'):
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'hospital'
        elif (row['PlaceType'] == 'guest_house') or (row['PlaceType'] == 'hostel') or (row['PlaceType'] == 'motel'):
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'hotel'
        elif (row['PlaceType'] == 'pub') or (row['PlaceType'] == 'nightclub'):
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'bar'
        elif (row['PlaceType'] == 'retail') or (row['PlaceType'] == 'shop') or (row['PlaceType'] == 'clothes'):
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'department_store'
        elif row['PlaceType'] == 'post_box':
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'post_office'
        elif (row['PlaceType'] == 'sports_centre') or (row['PlaceType'] == 'golf_course'):
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'sports'
        elif (row['PlaceType'] == 'atm') or (row['PlaceType'] == 'bank'):
            LocationEntity_withPlaceTypeData.loc[index, 'PlaceType'] = 'finance'
        
            
    #make subset of the dataset because some places are detected as not a place
    LocationEntity_withPlaceTypeData = LocationEntity_withPlaceTypeData.copy()
    LocationEntity_withPlaceTypeData = LocationEntity_withPlaceTypeData[LocationEntity_withPlaceTypeData['PlaceType'].
                                                                                      isin(['', 'city', 'city_district','water', 'path','bicycle', 'footway', 'pitch', 
                                                                                            'track', 'wood', 'address29', 'address26', 'address27',
                                                                                           'county', 'cycleway', 'hardware', 'furniture', 'computer',
                                                                                           'construction', 'information', 'industrial', 'jewelry',
                                                                                           'junction', 'locality', 'mobile_phone', 'nan', 'road', 'suburb',
                                                                                           'town', 'townhall', 'traffic_signals', 'tree', 'viewpoint',
                                                                                            'village', 'taxi', 'telephone']) == False]
    #get total number of places participant visit in a day
    unique_Location_Entity_withPlaceTypeData = LocationEntity_withPlaceTypeData.copy()
    unique_Location_Entity_withPlaceTypeData['timestamp'] = pd.to_datetime(pd.to_numeric(unique_Location_Entity_withPlaceTypeData['timestamp']), unit='ms')
    unique_Location_Entity_withPlaceTypeData.set_index('timestamp', drop=True, inplace=True)
    unique_Location_Entity_withPlaceTypeData = pd.DataFrame(unique_Location_Entity_withPlaceTypeData['PlaceType'])
    unique_Location_Entity_withPlaceTypeData = unique_Location_Entity_withPlaceTypeData.groupby(unique_Location_Entity_withPlaceTypeData.index.date)['PlaceType'].nunique()
    unique_Location_Entity_withPlaceTypeData = pd.DataFrame(unique_Location_Entity_withPlaceTypeData)
    unique_Location_Entity_withPlaceTypeData.rename(columns={'PlaceType':'total_places'}, inplace=True)
    
    extracted_LocationEntity_withPlaceTypeData = extract_features_categorical_data(LocationEntity_withPlaceTypeData.copy(), 'PlaceType', 24*60, 0.5)
    extracted_LocationEntity_withPlaceTypeData.drop(['Type'], axis=1, inplace=True)
    #resample per day 
    extracted_LocationEntity_withPlaceTypeData = resample_sum(extracted_LocationEntity_withPlaceTypeData.copy())
    extracted_LocationEntity_withPlaceTypeData = pd.concat([extracted_LocationEntity_withPlaceTypeData, unique_Location_Entity_withPlaceTypeData]).reset_index(drop=True)  
    extracted_LocationEntity_withPlaceTypeData = get_more_features(extracted_LocationEntity_withPlaceTypeData.copy())
    extracted_LocationEntity_withPlaceTypeData
    
    return extracted_LocationEntity_withPlaceTypeData
```
Extract time window features from DeviceEventEntity data

```
def DeviceEvent_extraction(DeviceEventEntityData): 
    ##make subset of the DeviceEventEntity data
    DeviceEventEntityData = DeviceEventEntityData[DeviceEventEntityData['type'].isin(['HEADSET_MIC_UNPLUGGED', 'TURN_OFF_DEVICE',
                                                      'HEADSET_MIC_UNPLUGGED', 'TURN_OFF_DEVICE']) == False]
    
    extracted_DeviceEventEntityData = extract_features_categorical_data(DeviceEventEntityData.copy(), 'type', 24*60, 0.5)
    extracted_DeviceEventEntityData.drop(['Type'], axis=1, inplace=True)
    extracted_DeviceEventEntityData = resample_sum(extracted_DeviceEventEntityData.copy())
    extracted_DeviceEventEntityData = get_more_features(extracted_DeviceEventEntityData.copy())
    
    return extracted_DeviceEventEntityData
```
Extract time window features from PhysicalActivityEntity data

```
def PhysicalActivityEvent_extraction(PhysicalActivityEventEntityData):
    
    ##make subset of the PhysicalActivityEventEntity data
    PhysicalActivityEventEntityData = PhysicalActivityEventEntityData[PhysicalActivityEventEntityData['confidence'] == '1.0']
    PhysicalActivityEventEntityData.drop(['confidence'], axis=1,inplace=True)
    PhysicalActivityEventEntityData.rename(columns={'type':'physicalActivity_type'}, inplace=True)
    
    extracted_PhysicalActivityEntityData = extract_features_categorical_data(PhysicalActivityEventEntityData.copy(), 'physicalActivity_type', 24*60, 0.5)
    extracted_PhysicalActivityEntityData.drop(['Type'], axis=1, inplace=True)
    extracted_PhysicalActivityEntityData = resample_sum(extracted_PhysicalActivityEntityData.copy())
    extracted_PhysicalActivityEntityData = get_more_features(extracted_PhysicalActivityEntityData.copy())
    
    return extracted_PhysicalActivityEntityData
```
Extract time window features from WifiEntity data

```
def Wifi_extraction (WifiEntityData):
    
    WifiEntityData.drop(['bssid', 'frequency', 'rssi'], axis=1,inplace=True)
    
    WifiEntityData['timestamp'] = pd.to_datetime(pd.to_numeric(WifiEntityData['timestamp']), unit='ms')
    WifiEntityData.set_index('timestamp', drop=True, inplace=True)
    extracted_WifiEntityData = WifiEntityData.groupby(WifiEntityData.index.date)['ssid'].nunique()
    extracted_WifiEntityData = pd.DataFrame(extracted_WifiEntityData)
    extracted_WifiEntityData = get_more_features(extracted_WifiEntityData.copy())
    
    return extracted_WifiEntityData
```
Extract time window features from ConnectivityEntity data

```
def Connectivity_extraction(ConnectivityEntityData):

    ##make subset of the ConnectivityEntity data
    ConnectivityEntityData = ConnectivityEntityData[ConnectivityEntityData['isConnected'].isin(['True'])]
    print(ConnectivityEntityData)
    print(ConnectivityEntityData['type'].unique())
    extracted_ConnectivityEntityData =  extract_features_categorical_data(ConnectivityEntityData.copy(), 'type', 60, 0.5)
    extracted_ConnectivityEntityData.drop(['Type'], axis=1, inplace=True)
    extracted_ConnectivityEntityData = resample_sum(extracted_ConnectivityEntityData.copy())
    extracted_ConnectivityEntityData = get_more_features(extracted_ConnectivityEntityData.copy())
    
    return extracted_ConnectivityEntityData
```

Extract time window features from DataTrafficEntity data (below is a function for extracting receiving data. there is transferring data extraction in seprate function but due to its similarity with receiving data, I will only add function for extracting receiving data)

```
def rx_extraction(rxTrafficData):
    rxTrafficData.drop(['duration'],axis=1, inplace=True)
    rxTrafficData['rxKiloBytes'] = pd.to_numeric(rxTrafficData['rxKiloBytes'])
    rxTrafficData = resample_sum(rxTrafficData.copy())
    
    extracted_rxTrafficData = get_more_features(rxTrafficData.copy())
    
    return extracted_rxTrafficData
```
Extract time window features from CallLogData data (below is a function for extracting outgoing calls data. the other call log status are in separate function but similar flow)

```
def CallLog_uniquenumber_ingoing_extraction(CallLogData_uniquenumber_ingoing):
    ##only get data whose type is ingoing
    CallLogData_uniquenumber_ingoing = CallLogData_uniquenumber_ingoing[CallLogData_uniquenumber_ingoing['type'] == 'INCOMING']
    CallLogData_uniquenumber_ingoing['timestamp'] = pd.to_datetime(pd.to_numeric(CallLogData_uniquenumber_ingoing['timestamp']), unit='ms')
    CallLogData_uniquenumber_ingoing.set_index('timestamp', drop=True, inplace=True)
    extracted_CallLogData_uniquenumber_ingoing = CallLogData_uniquenumber_ingoing.groupby(CallLogData_uniquenumber_ingoing.index.date)['number'].nunique()
    extracted_CallLogData_uniquenumber_ingoing = pd.DataFrame(extracted_CallLogData_uniquenumber_ingoing)
    extracted_CallLogData_uniquenumber_ingoing.rename(columns={'number': 'unique_number_ingoing'}, inplace=True)
    extracted_CallLogData_uniquenumber_ingoing = get_more_features(extracted_CallLogData_uniquenumber_ingoing.copy())
    
    return extracted_CallLogData_uniquenumber_ingoing
```
Extract time window features from MessageEntity data (below is a function for extracting inbox calmessagesls data. the other message status are in separate function but similar flow)

```
def Message_inbox_extraction(MessageData_inbox):
    MessageData_inbox.rename(columns={'messageBox': 'messageBox_inbox'}, inplace=True)
    MessageData_inbox = MessageData_inbox[MessageData_inbox['messageBox_inbox'].isin(['INBOX'])]
    MessageData_inbox['timestamp'] = pd.to_datetime(pd.to_numeric(MessageData_inbox['timestamp']), unit='ms')
    MessageData_inbox.set_index('timestamp', drop=True, inplace=True)
    extracted_MessageData_inbox = MessageData_inbox.groupby(MessageData_inbox.index.date)['messageBox_inbox'].count()
    extracted_MessageData_inbox = pd.DataFrame(extracted_MessageData_inbox, index=extracted_MessageData_inbox.index)
    extracted_MessageData_inbox = get_more_features(extracted_MessageData_inbox.copy())
    return extracted_MessageData_inbox
```
Extract time window features from Distance data

```
def Distance_extraction(DistanceData):
    #only take walking, jogging, running
    DistanceData = DistanceData[DistanceData['MotionType'].isin(['WALKING', 'JOGGING', 'RUNNING'])]
    DistanceData['timestamp'] = pd.to_datetime(pd.to_numeric(DistanceData['timestamp']), unit='ms')
    DistanceData.set_index('timestamp', drop=True, inplace=True)
    DistanceData['DistanceToday'] = pd.to_numeric(DistanceData['DistanceToday'])
    extracted_DistanceData = DistanceData.groupby(DistanceData.index.date)['DistanceToday'].max()
    extracted_DistanceData = pd.DataFrame(extracted_DistanceData)
    extracted_DistanceData = get_more_features(extracted_DistanceData.copy())
    return extracted_DistanceData
```
Extract time window features from Speed data

```
def Speed_extraction(SpeedData):
    SpeedData['timestamp'] = pd.to_datetime(pd.to_numeric(SpeedData['timestamp']), unit='ms')
    SpeedData.set_index('timestamp', drop=True, inplace=True)
    SpeedData['Speed'] = pd.to_numeric(SpeedData['Speed'])
    extracted_SpeedData = SpeedData.groupby(SpeedData.index.date)['Speed'].mean()
    extracted_SpeedData = pd.DataFrame(extracted_SpeedData)
    extracted_SpeedData = get_more_features(extracted_SpeedData.copy())
    return(extracted_SpeedData)
```

Extract time window features from Pedometer data

```
def Pedometer_extraction(PedometerData):
    PedometerData['timestamp'] = pd.to_datetime(pd.to_numeric(PedometerData['timestamp']), unit='ms')
    PedometerData.set_index('timestamp', drop=True, inplace=True)
    PedometerData['StepsToday'] = pd.to_numeric(PedometerData['StepsToday'])
    extracted_PedometerData = PedometerData.groupby(PedometerData.index.date)['StepsToday'].max()
    extracted_PedometerData = pd.DataFrame(extracted_PedometerData)
    extracted_PedometerData = get_more_features(extracted_PedometerData.copy())
    return extracted_PedometerData
```
Extract time window features from BatteryEntity data

```
def BatteryLevel_extraction(BatteryLevelData):
    BatteryLevelData['timestamp'] = pd.to_datetime(pd.to_numeric(BatteryLevelData['timestamp']), unit='ms')
    BatteryLevelData.set_index('timestamp', drop=True, inplace=True)
    BatteryLevelData['level'] = pd.to_numeric(BatteryLevelData['level'])
    extracted_BatteryLevelData =  BatteryLevelData.groupby(BatteryLevelData.index.date)['level'].mean()
    extracted_BatteryLevelData = pd.DataFrame(extracted_BatteryLevelData)
    extracted_BatteryLevelData = get_more_features(extracted_BatteryLevelData.copy())
    return extracted_BatteryLevelData
```
```
def BatteryStatus_extraction(BatteryStatusData):
    BatteryStatusData = extract_features_categorical_data(BatteryStatusData.copy(), 'status', 24*60, 0.5)
    BatteryStatusData.drop(['Type'], axis=1, inplace=True)
    extracted_BatteryStatusData = resample_sum(BatteryStatusData.copy())
    extracted_BatteryStatusData = get_more_features(extracted_BatteryStatusData.copy())
    return extracted_BatteryStatusData
```

After the features were extracted for each of participants sensor data, I did Min-Max scaling for all the features. This is to normalize the measurement for each features which were very contrast from each other. 

```
from sklearn.preprocessing import MinMaxScaler

def get_scaled_features(dataset):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    scaled = MinMaxScaler().fit_transform((dataset).to_numpy())
    dataset_scaled = pd.DataFrame(
      data=scaled[:,:],
      index = dataset.index,
      columns=dataset.columns
    )

    return dataset_scaled
```
I also removed the features that are filled less than 60%. <br/>

The final predictors / independent data looked like this <br/>
<img src="http://rafikatiwi.github.io/assets/sensor_data_project_predictors.png">


2. Predictive Model Building
Because there is imbalance in the personality dataset (personality dataset is user's self-report personality test result), I used SMOTE to augment the training dataset. 

```
from imblearn.over_sampling import SMOTE

def apply_SMOTE(x_data, y_data):
    sm = SMOTE(random_state=2)
    #get new training data
    X_train_res, y_train_res = sm.fit_sample(x_data, y_data)
    return X_train_res, y_train_res
```
```
from imblearn.over_sampling import SMOTENC
import pandas as pd
import numpy as np

K_FOLDS_RESAMPLE = []

for idx, (X_train, y_train, X_test, y_test) in enumerate(K_FOLDS):
  # categorical_features: masked arrays indicating where categorical feature is placed.
  #sampler = SMOTENC(categorical_features=FEATURE.columns.isin(CATEGORY))

  # 'fit_resample' conducts over-sampling data in the minority class.
  # Again, resampling should be only conducted in train set.
  X_sample, y_sample = apply_SMOTE(X_train, y_train)

  # Because SMOTENC.fit_resample() returns a tuple of numpy's array (not DataFrame or Series!),
  # We need to again build DataFrame and Series from resampled data.
  X_sample = pd.DataFrame(X_sample, columns=X_train.columns)
  y_sample = pd.Series(y_sample)

  K_FOLDS_RESAMPLE.append((X_sample, y_sample, X_test, y_test))
```

I built models using XGBoost and LightGBM (with the resampled training dataset) and compared the cross validated model performance. I set the baseline using Dummy Classifier
<img src="http://rafikatiwi.github.io/assets/sensor_data_project_modelevaluation.png"><br/>

Best performance for each personality type

Openness: XGBoost (0.58)<br/>
Conscientiousness: XGBoost (0.68)<br/>
Neuroticism: LightGBM (0.73)<br/>
Extraversion: LightGBM (0.71)<br/>
Agreeableness: LightGBM (0.64)<br/>
From the result, models to predict Neuroticism and Extraversion, respectively, performed the best

3. Feature Engineering

I used XGBoost to see feature importance to predict each type of personality

Openness Feature Importance
<br/>
<img src="http://rafikatiwi.github.io/assets/sensor_data_project_openness.png">
<br/>
We can see that the frequency of when user's battery status is full, and when the user's battery is disconnected from power, and user's visit to different places (university, hotel) are indicated as important features to predict openness.<br/>
<br/>
Conscientiousness Feature Importance
<br/>
<img src="http://rafikatiwi.github.io/assets/sensor_data_project_conscientiousness.png">
<br/>
We can see that the frequency of when user is inside a building (any kinds of building) and in a department store are indicated as important features to predict conscientiousness.<br/>
<br/>
Extraversion Feature Importance
<br/>
<img src="http://rafikatiwi.github.io/assets/sensor_data_project_extraversion.png">
<br/>
We can see that the frequency of when user is connected to different WiFi ssid and the average of their charging duration is indicated as important features to predict extraversion.<br/>
<br/>
Agreeableness Feature Importance
<br/>
<img src="http://rafikatiwi.github.io/assets/sensor_data_project_agreeableness.png">
<br/>
We can see that the the median duration of when user's battery status is still okay and the duration of user being inside building (any kinds of building) are indicated as important features to predict agreeableness.<br/>
<br/>
Neuroticism Feature Importance
<br/>
<img src="http://rafikatiwi.github.io/assets/sensor_data_project_neuroticism.png">
<br/>
We can see that the the frequency of a user being contacted by SMS, their okay battery status, and the steps taken are indicated as important features to predict neuroticism.<br/>
<br/>

Limitations:
1. User did not really move around much, most of the places are only in University. Making it harder to tell about their personality
