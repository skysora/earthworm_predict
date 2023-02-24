import PyEW
import time
from datetime import datetime
from collections import deque
#from staticmap import StaticMap, CircleMarker, Line
#import requests
import numpy as np
from dotenv import dotenv_values

def delaz(slon,slat,elon,elat):
    avlat=0.5*(elat+slat)
    a=1.840708+avlat*(.0015269+avlat*(-.00034+avlat*(1.02337e-6)))
    b=1.843404+avlat*(-6.93799e-5+avlat*(8.79993e-6+avlat*(-6.47527e-8)))
    dlat=slat-elat
    dlon=slon-elon
    dx=a*dlon*60.0
    dy=b*dlat*60.0
    delta=np.sqrt(dx*dx+dy*dy)
    
    return delta

try:
    env_config = dotenv_values(".env")
    p_survive = float(env_config["P_SURVIVE"])
    trig_dis = float(env_config["TRIG_DIS"])
    sta_count = int(env_config["STA_COUNT"])
    OUTPUT_MSG_TYPE = int(env_config["OUTPUT_MSG_TYPE"])


    mm = PyEW.EWModule(int(env_config["WAVE_RING_ID"]), int(env_config["PYEW_MODULE_ID"]), int(env_config["PYEW_INST_ID"]), 30.0, False)
    mm.add_ring(int(env_config["OUTPUT_RING_ID"]))

    wave_count = 0
    # flush EEW_RING
    while mm.get_bytes(0, OUTPUT_MSG_TYPE) != (0, 0):
        wave_count += 1
        continue
    print("EEW_RING flushed with " + str(wave_count) + " waves flushed")

    stalist = []
    # use a set to record non-repeat station
    existStation = set()
    lastSendTime = time.time()

    while True:
        pick_msg = mm.get_bytes(0, OUTPUT_MSG_TYPE)
        if pick_msg == (0, 0) and len(stalist) == 0:
            continue
        nowTime = time.time()
        if pick_msg != (0, 0):
            pick_str = pick_msg[2][:pick_msg[1]].decode("utf-8")
            if pick_str.split()[0] not in existStation:
                stalist.append((pick_str,nowTime))
                existStation.add(pick_str.split()[0])
                print(time.time(), pick_str)
        if nowTime - lastSendTime < p_survive:
            continue
        # filter repeated station
        for index,(sta,addTime) in enumerate(stalist):
            if nowTime - addTime > p_survive:
                stalist.pop(index)
                try:
                    existStation.remove(sta.split()[0])
                except ValueError:
                    pass
        # to prevent zero station after filter
        if len(stalist) == 0:
            continue
        # calculate avg lat, lon
        sum_lon = 0.0
        sum_lat = 0.0
        for (sta, addTime) in stalist:
            sum_lon += float(sta.split()[4])
            sum_lat += float(sta.split()[5])
        avg_lon = sum_lon/len(stalist)
        avg_lat = sum_lat/len(stalist)

        # associate
        match_sta = []
        for (sta, addTime) in stalist:
            dis = delaz(float(sta.split()[4]), float(sta.split()[5]), avg_lon, avg_lat)
            if dis < trig_dis:
                match_sta.append((sta, addTime))

        num_eew = 1
        nth=1
        Mpd = 5.0
        mark = env_config["REPORT_MARK"]
        n_phase = len(stalist)

        if(sta_count <= len(match_sta) < 10):
            # 6915 1638393844.124000 1638393834.420496 3 Mpd 5.7 24.56 121.46 10.0  7  6  5 0.0 0.0 2 147 9.7 fbh 0.7
            outmsg = f"{num_eew} {time.time()} {time.time()-3.0} {nth} Mpd {Mpd} {avg_lat:.2f} {avg_lon:.2f} {10} {n_phase} {n_phase} {n_phase} 0.5 0.5 1 100 10.0 {mark} 1.0 HWA {stalist}"
            print(outmsg)
            mm.put_msg(0,int(env_config["REPORT_MSG_TYPE"]),outmsg)
            lastSendTime = time.time()
except KeyboardInterrupt:
    mm.goodbye()


