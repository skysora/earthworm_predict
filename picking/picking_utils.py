import numpy as np
import time
import pandas as pd
import random
import torch
import requests
from datetime import datetime, timedelta
from staticmap import StaticMap, CircleMarker, Polygon, Line
from scipy import integrate
from math import sin, cos, sqrt, atan2, radians

# get the station's factor, latitude, lontitude, starttime, and endtime
def get_StationInfo(nsta_path, starttime):
    a_time = starttime[:4] + starttime[5:7] + starttime[8:10]
    st_time = int(a_time)
        
    d = {}
    with open(nsta_path) as f:
        for line in f.readlines():
            l = line.strip().split()
            
            # check the station is expired
            if st_time >= int(l[-2]) and st_time <= int(l[-1]):
                key = f"{l[0]}_{l[8]}_{l[7]}_0{l[5]}"
                d[key] = [float(l[1]), float(l[2]), [float(l[9]), float(l[10]), float(l[11])], l[-2], l[-1]]

    return d

# get the predicted station's latitude, lontitude, and factor
def get_coord_factor(key, stationInfo):    
    output = []
    sta = []
    for k in key:
        # 尋找相對應的測站
        tmp = k.split('_')
        if tmp[1][-1] == '1' or tmp[1][:2] == 'HL':
            channel = 'FBA'
        elif tmp[1][-1] == '4' or tmp[1][:2] == 'EH':
            channel = 'SP'
        elif tmp[1][-1] == '7' or tmp[1][:2] == 'HH':
            channel = 'BB'
            
        try:
            cur_k = f"{tmp[0]}_{channel}_{tmp[2]}_{tmp[3]}"
            sta.append(cur_k)
            info = stationInfo[cur_k]
            
            output.append(info[:-2])
        except:
            output.append([-1, -1, [1, 1, 1]])
        
    return output, sta

def get_PalertStationInfo(palert_path):
    df = pd.read_csv(palert_path)
    
    stationInfo = {}
    for i in df.iterrows():
        stationInfo[i[1]['station']] = [i[1]['lontitude'], i[1]['latitude'], 16.718]
    
    return stationInfo

def get_CWBStationInfo(cwb_path):
    with open(cwb_path, 'r') as f:
        sta_eew = f.readlines()
    
    stationInfo = {}
    for l in sta_eew:
        tmp = l.split(' ')
        
        stationInfo[tmp[0]] = [tmp[8], tmp[5], tmp[-2]]

    return stationInfo

def get_TSMIPStationInfo(tsmip_path):
    with open(tsmip_path, 'r') as f:
        sta_eew = f.readlines()

    stationInfo = {}
    for l in sta_eew:
        tmp = l.split()
        stationInfo[tmp[0]] = [tmp[5], tmp[4], tmp[-2]]

    return stationInfo

def get_Palert_CWB_coord(key, stationInfo):
    output = []
    station = []
    for k in key:
        sta = k.split('_')[0]
        print(f"sta:{sta}")
        output.append(stationInfo[sta])
        station.append(sta)
        
    return output, station

# load line tokens
def load_tokens(notify_path, waveform_path):
    with open(notify_path, 'r') as f:
        notify = f.readlines()

    with open(waveform_path, 'r') as f:
        waveform = f.readlines()

    notify = [n.strip() for n in notify]
    waveform = [n.strip() for n in waveform]
    return notify, waveform

# pick the p-wave according to the prediction of the model
def evaluation(pred, threshold_prob, threshold_trigger, threshold_type):
    # pred: 模型預測結果, (batch_size, wave_length)
    
    # 存每個測站是否 pick 到的結果 & pick 到的時間點
    pred_isTrigger = []
    pred_trigger_sample = []
    
    for i in range(pred.shape[0]):
        isTrigger = False
        
        if threshold_type == 'single':
            a = np.where(pred[i] >= threshold_prob, 1, 0)

            if np.any(a):
                c = np.where(a==1)
                isTrigger = True
                pred_trigger = c[0][0]
            else:
                pred_trigger = 0
                
        elif threshold_type == 'avg':
            a = pd.Series(pred[i])    
            win_avg = a.rolling(window=threshold_trigger).mean().to_numpy()

            c = np.where(win_avg >= threshold_prob, 1, 0)

            pred_trigger = 0
            if c.any():
                tri = np.where(c==1)
                pred_trigger = tri[0][0]-threshold_trigger+1
                isTrigger = True

        elif threshold_type == 'continue':
            a = np.where(pred[i] >= threshold_prob, 1, 0)
           
            a = pd.Series(a)    
            data = a.groupby(a.eq(0).cumsum()).cumsum().tolist()
          
            if threshold_trigger in data:
                pred_trigger = data.index(threshold_trigger)-threshold_trigger+1
                isTrigger = True
            else:
                pred_trigger = 0

        elif threshold_type == 'max':
            pred_trigger = np.argmax(pred[i]).item()
            
            if pred[i][pred_trigger] >= threshold_prob:
                isTrigger = True
            else:
                pred_trigger = 0

        # 將 threshold 與 picking time 分開進行
        if isTrigger:
            # 當 prediction 有過 threshold，則挑選最高的那個點當作 picking time
            pred_trigger = torch.argmax(pred[i]).item()
            
        pred_isTrigger.append(isTrigger)
        pred_trigger_sample.append(pred_trigger)
        
    return pred_isTrigger, pred_trigger_sample

def EEW_pick(res, pred_trigger):
    new_res = []
    for idx, pred in enumerate(pred_trigger):
        # 暴力篩選: 若是在 prediction window 前 50 個 samples picked，當作不算
        if pred <= 2500:
            new_res.append(False)
        else:
            new_res.append(True)

    return res, pred_trigger, new_res

# generate the Pa, Pv, Pd of picked stations
def picking_append_info(unnormed_wave, res, pred_trigger):
    sample_rate = 100.0
    upd = 300

    # unnormed_wave: acceleraion (batch, 3, wave_length)
    # taking only Z-axis

    # acc, vel, dis: (batch, wave_length)
    acc = unnormed_wave[:, 0]
    vel = integrate.cumtrapz(acc)/sample_rate
    dis = integrate.cumtrapz(vel)/sample_rate

    Pa, Pv, Pd = [], [], []
    for i in range(len(res)):
        # not picked
        if not res[i]:
            continue

        Pa.append(max(acc[i, pred_trigger[i]:pred_trigger[i]+upd]))
        Pv.append(max(vel[i, pred_trigger[i]:pred_trigger[i]+upd]))
        Pd.append(max(dis[i, pred_trigger[i]:pred_trigger[i]+upd]))

    return Pa, Pv, Pd

# generate the p's weight
def picking_p_weight_info(pred, res):
    p_weight = []
    for i in range(len(res)):
        # not picked
        if not res[i]:
            continue

        peak = torch.max(pred[i]).item()

        if peak >= 0.7:
            weight = 0
        elif peak >= 0.6 and peak < 0.7:
            weight = 1
        elif peak >= 0.5 and peak < 0.6:
            weight = 2
        else:
            weight = 3

        p_weight.append(weight)

    return p_weight

# Calculate the distance between two coordinates
def distance(coord1, coord2):
    R = 6373.0

    lon1, lat1 = radians(coord1[0]), radians(coord1[1])
    lon2, lat2 = radians(coord2[0]), radians(coord2[1])

    dlon, dlat = lon2-lon1, lat2-lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    distance = R * c
    return distance

# 區域型 picking
def post_picking(station_factor_coords, res, threshold_km):
    pick_idx = np.arange(len(res))[res]

    total_lon, total_lat = 0.0, 0.0
    for idx, pick_sta in enumerate(pick_idx):
        total_lon += station_factor_coords[pick_sta][0]
        total_lat += station_factor_coords[pick_sta][1]

    avg_lon, avg_lat = total_lon / len(pick_idx), total_lat / len(pick_idx)
    middle_coord = (avg_lon, avg_lat)

    real_picked_res = []
    for idx, pick_result in enumerate(res):
        if not pick_result:
            real_picked_res.append(False)
            continue

        dis = distance(middle_coord, (station_factor_coords[idx][0], station_factor_coords[idx][1]))

        if dis <= threshold_km:
            real_picked_res.append(True)
        else:
            real_picked_res.append(False)

    return real_picked_res

# 鄰近測站 picking 
def neighbor_picking(neighbor_table, station_list, res, threshold_neighbor):
    pick_sta = np.array(station_list)[res]
    real_res = []
    
    for idx, sta in enumerate(station_list):
        if not res[idx]:
            real_res.append(False)
            continue
            
        n_pick = len(set(pick_sta).intersection(set(neighbor_table[sta])))

        if (len(neighbor_table[sta]) < threshold_neighbor and n_pick > 1) or (n_pick >= threshold_neighbor):
            real_res.append(True)
        else:
            real_res.append(False)
            
    return real_res

# 建表: 先把每個測站方圓 threshold_km 內的所有測站都蒐集進 dict
def build_neighborStation_table(stationInfo, threshold_km=None, nearest_station=3, option='nearest'):
    table = {}
    
    for sta in stationInfo:
        table[sta[0]] = []
    
    if option == 'km':
        for outer_idx, sta1 in enumerate(stationInfo):
            nearest3_sta, nearest3_dis = [], []
            for inner_idx, sta2 in enumerate(stationInfo):
                if inner_idx <= outer_idx:
                    continue

                dis = distance((sta1[1][0], sta1[1][1]), (sta2[1][0], sta2[1][1]))
                if dis <= threshold_km:
                    table[sta1[0]].append(sta2[0])
                    table[sta2[0]].append(sta1[0])
    
    elif option == 'nearest':
        for outer_idx, sta1 in enumerate(stationInfo):
            nearest3_sta, nearest3_dis = [], []
            for inner_idx, sta2 in enumerate(stationInfo):
                if sta1[0] == sta2[0]:
                    continue
                    
                dis = distance((sta1[1][0], sta1[1][1]), (sta2[1][0], sta2[1][1]))
                nearest3_sta.append(sta2[0])
                nearest3_dis.append(dis)

            nearest3_sta = np.array(nearest3_sta)
            nearest3_dis = np.array(nearest3_dis)
            table[sta1[0]] = nearest3_sta[np.argsort(nearest3_dis)[:nearest_station]].tolist()
            
    return table

# 檢查 picked station 在 picktime_gap 秒內有沒有 pick 過了，有的話先 ignore
def check_duplicate_pick(res, toPredict_scnl, pick_record, pred_trigger, waveform_starttime, pick_gap):
    pick_idx = np.arange(len(res))[res]
    sampling_rate = 100.0

    # 每次 picking 都會有誤差，跟上次 picking time 要差距超過 error_threshold 秒才能算是新地震
    error_threshold_sec = pick_gap
    for i in pick_idx:
        # convert pred_trigger into absolute datetime
        pick_sample = pred_trigger[i]
        sec, microsec = pick_sample // sampling_rate, pick_sample % sampling_rate
        
        # in picker, we take last 3000 samples waveforms
        # +8 hours is because .timestamp() will minus 8 hours automatically
        p_arrival_time = waveform_starttime + timedelta(seconds=float(sec), microseconds=float(microsec)*10000, hours=8)
        
        # if the station is picked in first time
        if toPredict_scnl[i] not in pick_record:
            pick_record[toPredict_scnl[i]] = p_arrival_time
        # if the station is picked before, check the previous picking time is PICK_TIME_GAP seconds ago
        else:
            prev_pickingtime = pick_record[toPredict_scnl[i]]
            
            if (p_arrival_time > prev_pickingtime) and (p_arrival_time - prev_pickingtime).seconds >= error_threshold_sec:
                pick_record[toPredict_scnl[i]] = p_arrival_time
            else:
                res[i] = False
            
    return res, pick_record

# generate the picking message 
# H024 HLZ TW -- 120.204309 22.990450 2.549266 0.059743 0.039478 0.000000(quality) 1671078524.54000 2(waveform quality) 1 3
def gen_pickmsg(station_factor_coords, res, pred_trigger, toPredict_scnl, waveform_starttime, Pavd, P_weight, sampling_rate=100):
    pick_msg = []
    
    # grep the idx of station which is picked
    pick_idx = np.arange(len(res))[res]
   
    for idx, pick_sta in enumerate(pick_idx):
        # station info is unknown
        if station_factor_coords[pick_sta][0] == -1:
            continue
            
        cur_pickmsg = ""

        # append scnl into pick_msg
        scnl = toPredict_scnl[pick_sta].split('_')
        
        # convert channel: 1-9 to HL (1-3), EH (4-6), HH (7-9)
        if scnl[1][-1] == '1':
            channel = 'HLZ'
        elif scnl[1][-1] == '4':
            channel = 'EHZ'
        elif scnl[1][-1] == '7':
            channel = 'HHZ'
        else:
            channel = scnl[1]
        cur_pickmsg += f"{scnl[0]} {channel} {scnl[2]} {scnl[3]}"
        
        # append coordinate into pick_msg
        cur_pickmsg += f" {station_factor_coords[pick_sta][0]} {station_factor_coords[pick_sta][1]}"

        # Pa, Pv, and Pd
        # Tc is useless, fixed at 0.0
        cur_pickmsg += f" {Pavd[0][idx]} {Pavd[1][idx]} {Pavd[2][idx]} 0.0"
        
        # append p_arrival time into pick_msg
        pick_sample = pred_trigger[pick_sta]
        sec, microsec = pick_sample // sampling_rate, pick_sample % sampling_rate
        
        # in picker, we take last 3000 samples waveforms
        # +8 hours is because .timestamp() will minus 8 hours automatically
        p_arrival_time = waveform_starttime + timedelta(seconds=float(sec), microseconds=float(microsec)*10000, hours=8)
        cur_pickmsg += f' {p_arrival_time.timestamp()}'

        # p_weight
        cur_pickmsg += f' {P_weight[idx]}'

        # fixed instrument with 1: acceleration
        # fixed upd_sec with 3: always use 3 seconds after p_arrival to calculate the Pa, Pv, and Pd
        cur_pickmsg += ' 1 3'
        cur_pickmsg += ' AI_picker'
     
        pick_msg.append(cur_pickmsg)
        
    return pick_msg

# plot the picking info on Taiwan map
def plot_taiwan(name, coords1, token, token_number):
    m = StaticMap(300, 400)

    for sta in coords1:
        marker = CircleMarker(sta, '#eb4034', 8)

        m.add_marker(marker)

    image = m.render(zoom=7)
    image.save(f"./plot/trigger/{name}.png")

    token_number = random.sample(range(len(token)), k=1)[0]
    token_number = notify(len(coords1), name, token, token_number)
    
    return token_number

# send the picking info to Line notify
def notify(n_sta, name, token, token_number):
    message = str(n_sta) + ' 個測站偵測到 P 波\n報告時間: '+str((datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f'))
    message += '\n'

    while True:
        if token_number >= len(token):
            token_number = len(token) - 1
            
        try:
            url = "https://notify-api.line.me/api/notify"
            headers = {
                'Authorization': f'Bearer {token[token_number]}'
            }
            payload = {
                'message': message,
            }
            image = {
                'imageFile': open(f"./plot/trigger/{name}.png", 'rb'),
            }
            response = requests.request(
                "POST",
                url,
                headers=headers,
                data=payload,
                files=image,
            )
            if response.status_code == 200:
                print(f"Success coordination -> {response.text}")
                break
            else:
                print(f'(Notify) Error -> {response.status_code}, {response.text}')
                token_number = random.sample(range(len(token)), k=1)[0]
        except Exception as e:
            print(e)
            token_number = random.sample(range(len(token)), k=1)[0]
    return token_number

# send the picking info to Line notify
def plot_notify(name, token, token_number):    
    msg = name.split('/')[-1].split('.')[0] 
    message = f"Prediction: {msg}\n"

    while True:
        if token_number >= len(token):
            token_number = len(token) - 1
            
        try:
            url = "https://notify-api.line.me/api/notify"
            headers = {
                'Authorization': f'Bearer {token[token_number]}'
            }
            payload = {
                'message': message,
            }
            image = {
                'imageFile': open(name, 'rb'),
            }
            response = requests.request(
                "POST",
                url,
                headers=headers,
                data=payload,
                files=image,
            )
            if response.status_code == 200:
                print(f"Success waveform prediction -> {response.text}")
                break
            else:
                print(f'(Waveform) Error -> {response.status_code}, {response.text}')
                token_number = random.sample(range(len(token)), k=1)[0]
        except Exception as e:
            print(e)
            token_number = random.sample(range(len(token)), k=1)[0]
    return token_number

# sent the notify proved the system is alive
def alive_notify(token, token_number):
    message = 'System is alive: \n'
    message += (datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f')

    while True:
        if token_number >= len(token):
            token_number = len(token) - 1
            
        try:
            url = "https://notify-api.line.me/api/notify"
            headers = {
                'Authorization': f'Bearer {token[token_number]}'
            }
            payload = {
                'message': message,
            }
            response = requests.request(
                "POST",
                url,
                headers=headers,
                data=payload,
            )
            if response.status_code == 200:
                print(f"Success, system is alive -> {response.text}")
                break
            else:
                print(f'(Alive) Error -> {response.status_code}, {response.text}')
                token_number = random.sample(range(len(token)), k=1)[0]
        except Exception as e:
            print(e)
            token_number = random.sample(range(len(token)), k=1)[0]
    return token_number
