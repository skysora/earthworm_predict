import numpy as np
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from staticmap import StaticMap, CircleMarker, Polygon, Line
from scipy import integrate

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
                d[key] = [float(l[1]), float(l[2]),float(l[3]), [float(l[9]), float(l[10]), float(l[11])], l[-2], l[-1]]

    return d

# get the predicted station's latitude, lontitude, and factor
def get_coord_factor(key, stationInfo):    
    output = []
    for k in key:
        # 尋找相對應的測站
        tmp = k.split('_')
        if tmp[1][-1] == '1' or tmp[1][:2] == 'HL':
            channel = 'FBA'
        elif tmp[1][-1] == '4' or tmp[1][:2] == 'EH':
            channel = 'SP'
        elif tmp[1][-1] == '7' or tmp[1][:2] == 'HH':
            channel = 'BB'
        
        cur_k = f"{tmp[0]}_{channel}_{tmp[2]}_{tmp[3]}"
        
        try:
            info = stationInfo[cur_k]
            output.append(info[:-2])
        except:
            output.append([-1, -1,-1, [1, 1, 1]])
        
    return output

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
    for k in key:
        output.append(stationInfo[k.split('_')[0]])
        
    return output

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

        pred_isTrigger.append(isTrigger)
        pred_trigger_sample.append(pred_trigger)
        
    return pred_isTrigger, pred_trigger_sample

# generate the Pa, Pv, Pd of picked stations
def picking_append_info(unnormed_wave, res, pred_trigger):
    sample_rate = 100.0
    
    # unnormed_wave: acceleraion (batch, 3, wave_length)
    acc = np.sqrt(unnormed_wave[:, 0]**2+unnormed_wave[:, 1]**2+unnormed_wave[:, 2]**2)

    vel = integrate.cumtrapz(acc)/sample_rate

    dis = integrate.cumtrapz(vel)/sample_rate

    # Pa
    Pa = acc[:, pred_trigger][0][res]

    # Pv
    Pv = vel[:, pred_trigger][0][res]

    # Pd
    Pd = dis[:, pred_trigger][0][res]

    return Pa, Pv, Pd

# generate the picking message 
# H024 HLZ TW -- 120.204309 22.990450 2.549266 0.059743 0.039478 0.000000(quality) 1671078524.54000 2(waveform quality) 1 3
def gen_pickmsg(station_factor_coords, res, pred_trigger, toPredict_scnl, waveform_starttime, Pavd, sampling_rate=100):
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
        
        # append useless info and quality into pick_msg
        # TODO: Tc
        cur_pickmsg += f" {Pavd[0][idx]} {Pavd[1][idx]} {Pavd[2][idx]} 0.0"
        
        # append p_arrival time into pick_msg
        pick_sample = pred_trigger[pick_sta]
        sec, microsec = pick_sample // sampling_rate, pick_sample % sampling_rate
        
        # in picker, we take last 3000 samples waveforms
        # +8 hours is because .timestamp() will minus 8 hours automatically
        p_arrival_time = waveform_starttime + timedelta(seconds=float(sec), microseconds=float(microsec)*10000, hours=8)
        cur_pickmsg += f' {p_arrival_time.timestamp()}'
        
        # append waveform's quality and useless info into pick_msg
        cur_pickmsg += ' 2 1 3'
     
        pick_msg.append(cur_pickmsg)
        
    return pick_msg

# plot the picking info on Taiwan map
def plot_taiwan(name, coords1):
    m = StaticMap(300, 400)

    for sta in coords1:
        # if sta[0] < 120 or sta[1] < 23 or sta[0] > 122 or sta[1] > 25:
        #     continue
        #marker_outline = CircleMarker(sta, 'white', 18)

        marker = CircleMarker(sta, '#eb4034', 8)

        #m.add_marker(marker_outline)
        m.add_marker(marker)

    image = m.render(zoom=7)
    image.save(f"./plot/trigger/{name}.png")

    notify(len(coords1), name)

# send the picking info to Line notify
def notify(n_sta, name):
    token = "ww5NO5lx7VcqvbqZY4YiLYbY7Caujg9hVieiiTXyXvl"

    message = str(n_sta) + ' 個測站偵測到 P 波\n報告時間: '+str((datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f'))
    message += '\n'

    try:
        url = "https://notify-api.line.me/api/notify"
        headers = {
            'Authorization': f'Bearer {token}'
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
    except Exception as e:
        print(e)

# send the picking info to Line notify
def plot_notify(name):
    token = "u2vxxmhVBnUAMFWFfYhcku60fxj2vpR2i2N1QtkffDL"
    
    msg = name.split('/')[-1].split('.')[0] 
    message = f"Prediction: {msg}\n"

    try:
        url = "https://notify-api.line.me/api/notify"
        headers = {
            'Authorization': f'Bearer {token}'
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
    except Exception as e:
        print(e)

