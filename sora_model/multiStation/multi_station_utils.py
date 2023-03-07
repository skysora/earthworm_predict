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
  
  
def plot_wave(waves,name):
    
    # fig, ax = plt.subplots(5,5,constrained_layout = True,figsize=(20, 10))
    # for i in range(1,6):
    #     for j in range(1,6):
    #         index = (i*j)-1
    #         # wave = np.sqrt(waves[0,index,:,0]**2+waves[0,index,:,1]**2+waves[0,index,:,2]**2)
    #         wave = waves[0,index,0,:]
    #         ax[i-1,j-1].plot(wave)
    # plt.savefig(f'{name}')
    # plt.close()
    fig, ax = plt.subplots(5,5,constrained_layout = True,figsize=(20, 10))
    for i in range(1,6):
        for j in range(1,6):
            index = (i*j)-1
            # wave = np.sqrt(waves[index,:,0]**2+waves[index,:,1]**2+waves[index,:,2]**2)
            wave = waves[0,index,0,:]
            ax[i-1,j-1].plot(wave)
    plt.savefig(f'{name}')
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