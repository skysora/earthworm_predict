#lat 緯度
#lon 經度
from staticmap import StaticMap, CircleMarker, Polygon, Line
import requests
import matplotlib.pyplot as plt      # v 3.3.2
import numpy as np

def location_transformation(metadata, target=None):
    transform_target_only = False
    scale_metadata = True
    pos_offset = [0,12]
    D2KM = D2KM = 111.19492664455874
    metadata = metadata.copy()

    metadata_old = metadata
    metadata = metadata.copy()
    mask = (metadata == 0).all(axis=2)
    if target is not None:
        target[:, 0] -= pos_offset[0]
        target[:, 1] -= pos_offset[1]
    metadata[:, :, 0] -= pos_offset[0]
    metadata[:, :, 1] -= pos_offset[1]

    # Coordinates to kilometers (assuming a flat earth, which is okay close to equator)
    if scale_metadata:
        metadata[:, :, :2] *= D2KM
    if target is not None:
        target[:, :2] *= D2KM

    metadata[mask] = 0

    if scale_metadata:
        metadata /= 100
    if target is not None:
        target /= 100

    if transform_target_only:
        metadata = metadata_old

    if target is None:
        return metadata
    else:
        return metadata, target


def plot_taiwan(target_city,name):
    
    m = StaticMap(1000, 1000)

    for index in target_city:
    
        sta = target_city[index]
        try:
            max_pga_level = [index for (index, item) in enumerate(sta[-1]) if item == 1][-1]
        except:
            max_pga_level=999
            
        if(max_pga_level!=999):
            # print("Draw")
            if(max_pga_level==0):
                color="#00FFFF"
            elif(max_pga_level==1):
                color="#0000FF"
            elif(max_pga_level==2):
                color='#FFFF00'
            else:
                color="#E60000"
            
            marker = CircleMarker([sta[1], sta[2]], color, 10) 
            m.add_marker(marker)
        
    
    m.add_marker(CircleMarker((120, 25), '#00000000', 12))
    m.add_marker(CircleMarker((123, 25), '#00000000', 12))
    m.add_marker(CircleMarker((120, 22), '#00000000', 12))
    m.add_marker(CircleMarker((123, 22), '#00000000', 12))

    image = m.render(zoom=8)
    image.save(f'{name}')
  
  
def plot_wave(waves,position_list):
    print(position_list.keys())
    for i in range(250):
        if(i not in position_list.keys()):
            continue
        print(i)
        plt.plot(waves[0,i,0,:])
        plt.savefig(f'{i}.png')
        plt.clf()
        plt.close()
      
# send the picking info to Line notify
def multi_station_plot_notify(name):
    token = "uAUGiQLwsDHPjahFHAPWEmTztOFipJIB4O8bmhaFlLm"
    
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
        
# send the picking info to Line notify
def multi_station_msg_notify(msg):
    token = "EDHcaMNDRH3xyHxpKrBaYtPUGjZjh2xCTyDJmkws0cR"
    
    message = f"Prediction: {msg}\n"

    try:
        url = "https://notify-api.line.me/api/notify"
        headers = {
            'Authorization': f'Bearer {token}'
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
            print(f"Success waveform prediction -> {response.text}")
    except Exception as e:
        print(e)
        
        
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

def mapping_gal_to_intensity(gal):
    if gal == '8gal':
        return '3級'
    if gal == '25gal':
        return '4級'
    if gal == '81gal':
        return '5弱級'
    if gal == '140gal':
        return '5強級'
    if gal == '250gal':
        return '6弱級'
    
def gen_info(df):
    # output info
    result = {}

    true_intensity_level = ['1', '2', '3', '4', '5弱', '5強', '6弱']
    true_intensity_key = ['Label_0.8gal', 'Label_2.5gal', 'Label_8gal', 'Label_25gal', 'Label_81gal', 'Label_140gal', 'Label_250gal']
    pred_intensity_level = ['3', '4', '5弱', '5強', '6弱']
    pred_intensity_key = ['8gal', '25gal', '81gal', '140gal', '250gal']
    
    for data in df.iterrows():
        tmp_station = data[1]['County'] + ' ' + data[1]['Township'] + ' ' + data[1]['Station_Chinese_Name']
        if tmp_station not in result.keys():
            result[tmp_station] = {}

        gt = np.array([data[1][k] for k in true_intensity_key])
        pred = np.array([data[1][k] for k in pred_intensity_key])
        
        
        mask_gt = np.logical_or(gt == '0', gt == '0.0')
        mask_pred = pred == 1

        
        # 檢查真實情況有沒有超過 1 級
        if np.any(mask_gt):
            gt_intensity = mask_gt.tolist().count(False)
            result[tmp_station]['true_intensity'] = true_intensity_level[gt_intensity-1]
        else:
            result[tmp_station]['true_intensity'] = 0

        # 檢查預測情況有沒有超過 1 級
        if np.any(mask_pred):
            prediction_intensity = mask_pred.tolist().count(True)
            result[tmp_station]['pred_intensity'] = pred_intensity_level[prediction_intensity-1]
        else:
            result[tmp_station]['pred_intensity'] = 0

        
        # 計算 leading time
        for idx, p_inten in enumerate(mask_pred):
            if p_inten == False:
                continue
            
            level = true_intensity_key[idx+2]
            pred_time = data[1]['Warning_Time']
            gt_time = data[1][level]
            
            if 'time_diff' not in result[tmp_station]:
                result[tmp_station]['time_diff'] = {}
            
            if gt_time == 0:
                result[tmp_station]['time_diff'][level] = 'false positive'
            else:
                gt_time = datetime.strptime(gt_time, '%Y-%m-%d %H:%M:%S.%f')
                pred_time = datetime.strptime(pred_time, '%Y-%m-%d %H:%M:%S.%f')
                if pred_time < gt_time:
                    time_diff = gt_time - pred_time
                    time_diff = f"提早 {round(time_diff.seconds + time_diff.microseconds/10000 / 100, 2)} 秒"
                else:
                    time_diff = pred_time - gt_time
                    time_diff = f"晚了 {round(time_diff.seconds + time_diff.microseconds/10000 / 100, 2)} 秒"
                    
                result[tmp_station]['time_diff'][level] = time_diff
        
        if(int(result[tmp_station]["true_intensity"][0])>=0 and int(str(result[tmp_station]["pred_intensity"])[0])==0):
            result[tmp_station]['time_diff'] = {}
            for idx, p_inten in enumerate(mask_gt):
                if p_inten == False:
                    continue 
                level = true_intensity_key[idx]
                # result[tmp_station]['time_diff'][level] = 'true negative'
    return result

def send_info(df):
    result = gen_info(df)
    cnt = 0
    msg = ""
    for k, v in result.items():
        msg += f"\n{k} \n預測: {v['pred_intensity']}級, 真實級別: {v['true_intensity']}級\n"
        msg += f"|\t震度\t|\tleading time\t|\n"

        for tmp_k, tmp_v in v['time_diff'].items():
            msg += f"|\t{mapping_gal_to_intensity(tmp_k.split('_')[-1])}\t|\t{tmp_v}\t|\n"

        msg += '=-=-=-=-=-=-=-=-'
        cnt += 1

        if cnt % 5 == 0:
            # alive_notify(msg)
            msg = ""

    return msg

# sent the notify proved the system is alive
def alive_notify(msg):
    message = '報告時間: \n' + (datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f') + '\n'
    message += msg

    token = 'gPDTduLxtElER8j4T2glCQXh3vRJZTtSlKjvhfDaCJb'
    while True:
        try:
            url = "https://notify-api.line.me/api/notify"
            headers = {
                'Authorization': f'Bearer {token}'
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
        except Exception as e:
            print(e)

def output_msg(df_path):
    df = pd.read_csv(df_path)
    true_intensity_key = ['Label_0.8gal', 'Label_2.5gal', 'Label_8gal', 'Label_25gal', 'Label_81gal', 'Label_140gal', 'Label_250gal']
    for key in true_intensity_key:
        df[key] = df[key].fillna(0)
    msg = send_info(df)
    return msg


