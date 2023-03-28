import time
import numpy as np
import torch
import torch.nn.functional as F

import multiprocessing
from multiprocessing import Process, Manager, Array, Value, Queue
from itertools import compress
import threading
import torch.multiprocessing
from torch.multiprocessing import Process, Manager, Array, Value, Queue
import PyEW
import ctypes as c
import random
import pandas as pd 
import os
import sys
import bisect
from scipy.stats import norm
from dotenv import dotenv_values
sys.path.append('../')
from tqdm import tqdm
from ctypes import c_char_p
import glob
from datetime import datetime, timedelta
from collections import deque
from picking_preprocess import *
from picking_utils import *
import json
import csv


#multi-station

import multi_station_warning.models as models
from multi_station_warning.util import *

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/TEAM/lib
# CUDA_VISIBLE_DEVICES=2 python3 EEW.py




# select the stations to collect the waveform
def station_selection(sel_chunk, station_list, opt, n_stations=None):
    if opt == 'CWB':
        lon_split = np.array([120.91])
        lat_split = np.array([21.913 , 22.2508961, 22.5887922, 22.9266883, 23.2645844,
        23.6024805, 23.9403766, 24.2782727, 24.6161688, 24.9540649,
        25.291961 ])
        
        # 依照經緯度切分測站
        chunk = [[] for _ in range(40)]
        for k, sta in station_list.items():
            row, col = 0, 0

            row = bisect.bisect_left(lon_split, float(sta[0]))
            col = bisect.bisect_left(lat_split, float(sta[1]))
            
            chunk[2*col+row].append((k, [float(sta[0]), float(sta[1]), float(sta[2])]))

        # 微調前步驟的結果，使每個區域都不超過 55 個測站
        output_chunks = []
        output_chunks.append(chunk[5]+chunk[4]+chunk[3]+chunk[2]+chunk[0])
        output_chunks.append(chunk[6])
        output_chunks.append(chunk[7]+chunk[9]+chunk[11])
        output_chunks.append(chunk[13]+chunk[15])
        output_chunks.append(chunk[14])
        output_chunks.append(chunk[16]+chunk[18])
        output_chunks.append(chunk[17])
        
        chunk[19] = sorted(chunk[19], key = lambda x : x[1][1])
        output_chunks.append(chunk[19][len(chunk[19])//2:])
        output_chunks.append(chunk[19][:len(chunk[19])//2])
        
        tmp_chunk = chunk[8]+chunk[10]+chunk[12]
        tmp_chunk = sorted(tmp_chunk, key = lambda x : x[1][1])
        output_chunks.append(tmp_chunk[:len(tmp_chunk)//4])
        output_chunks.append(tmp_chunk[len(tmp_chunk)//4:len(tmp_chunk)//4 * 2])
        output_chunks.append(tmp_chunk[len(tmp_chunk)//4 * 2:len(tmp_chunk)//4 * 3])
        output_chunks.append(tmp_chunk[len(tmp_chunk)//4 * 3:])
        
        chunk[21] = sorted(chunk[21], key = lambda x : x[1][1])
        output_chunks.append(chunk[21][len(chunk[21])//3:len(chunk[21])//3 * 2])
        output_chunks.append(chunk[21][len(chunk[21])//3 * 2:len(chunk[21])//3 * 3])
        output_chunks.append(chunk[21][len(chunk[21])//3 * 3:])
    
        output = []
        for o in output_chunks[sel_chunk]:
            output.append(o[0])

        return output
    elif opt == 'TSMIP':
        # lon_split = np.array([120.91])
        # lat_split = np.array([21.9009 , 24.03495, 26.169  ])

        # # 依照經緯度切分測站
        # chunk = [[] for _ in range(6)]
        # for k, sta in station_list.items():
        #     row, col = 0, 0

        #     row = bisect.bisect_left(lon_split, float(sta[0]))
        #     col = bisect.bisect_left(lat_split, float(sta[1]))
            
        #     chunk[2*col+row].append((k, [float(sta[0]), float(sta[1]), float(sta[2])]))

        # output_chunks = []
        # output_chunks.append(chunk[3])
        
        # chunk[2] = sorted(chunk[2], key = lambda x : x[1][1])
        # output_chunks.append(chunk[2][len(chunk[2])//2:])
        # output_chunks.append(chunk[2][:len(chunk[2])//2])
        # output_chunks[-1] += chunk[0]
        
        # chunk[5] = sorted(chunk[5], key = lambda x : x[1][0])
        # output_chunks.append(chunk[5][:50] + chunk[4])
        # output_chunks.append(chunk[5][50:])

        # output = []
        # for o in output_chunks[sel_chunk]:
        #     output.append(o[0])

        # return output
        output = []
        for k, sta in station_list.items():
            output.append(k)

        return output
    else:
        # sort the station_list by lontitude
        stationInfo = sorted(station_list.items(), key = lambda x : x[1][0])

        station_chunks = [stationInfo[n_stations*i:n_stations*i+n_stations] 
                            for i in range(len(stationInfo)//n_stations)]

        return [i[0] for i in station_chunks[sel_chunk]]

# to save wave form from WAVE_RING
def WaveSaver(env_config, waveform_buffer, key_index, nowtime, waveform_buffer_start_time, key_cnt, stationInfo):
    MyModule = PyEW.EWModule(int(env_config["WAVE_RING_ID"]), int(env_config["PYEW_MODULE_ID"]), int(env_config["PYEW_INST_ID"]), 30.0, False)
    MyModule.add_ring(int(env_config["WAVE_RING_ID"])) # WAVE_RING
    MyModule.add_ring(int(env_config["PICK_RING_ID"])) # PICK_RING
    # MyModule.add_ring(int(env_config["OUTPUT_RING_ID"])) # OUTPUT_RING
    wave_count = 0
    # flush WAVE_RING
    while MyModule.get_wave(0) != {}:
        wave_count += 1
        continue
    print("WAVE_RING flushed with " + str(wave_count) + " waves flushed")

    wave_count = 0
    # flush PICK_RING
    while MyModule.get_bytes(1, int(env_config["PICK_MSG_TYPE"])) != (0, 0):
        wave_count += 1
        continue
    print("PICK_RING flushed with " + str(wave_count) + " waves flushed")

    # partial_station_list = []
    # if env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'TSMIP':
    #     partial_station_list = station_selection(int(env_config["CHUNK"]), stationInfo, env_config['SOURCE'], int(env_config["N_PREDICTION_STATION"]))
    # print(partial_station_list)
    while True:
        # get raw waveform from WAVE_RING
        wave = MyModule.get_wave(0) 
        
        # keep getting wave until the wave isn't empty
        # if wave != {}:
                # print(wave['station'] not in partial_station_list)
        # if wave == {} or (wave['station'] not in partial_station_list and (env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'TSMIP')):
        if wave == {}:    
            continue
        
        station = wave['station']
        channel = wave['channel']
        network = wave['network']
        location = wave['location']
        startt = wave['startt']
        nsamp = wave['nsamp']
        scnl = f"{station}_{channel}_{network}_{location}"
        # print(f"start processing: {snl}")
        
        # The input scnl hasn't been saved before
        if scnl not in key_index:
            # save it into a dict which saves the pair of scnl and waveform's index
            key_index[scnl] = int(key_cnt.value)
            # initialize the scnl's waveform with shape of (1,6000) and fill it with 0
            # waveform_buffer[int(key_cnt.value)] = np.zeros((1, 6000))
            waveform_buffer[int(key_cnt.value)] = torch.zeros((1, 6000))
            # waveform_buffer = np.append(waveform_buffer, np.zeros((1,6000)), axis = 0)
            key_cnt.value += 1

        # find input time's index to save it into waveform accouding this index
        startIndex = int(startt*100) - int(waveform_buffer_start_time.value)
        if startIndex < 0:
            # the wave is too old to save into buffer
            wave = MyModule.get_wave(0)
            continue
            
        # save wave into waveform from starttIndex to its wave length
        try:
            # print(key_index[scnl], startIndex, startIndex+nsamp)
            # print(waveform_buffer.shape)
            waveform_buffer[key_index[scnl]][startIndex:startIndex+nsamp] = torch.from_numpy(wave['data'].copy().astype(np.float32))
        except Exception as e:
            print(e)
            print(f"{scnl} can't assign wave data into waveform")
            print(key_index[scnl], startIndex, startIndex+nsamp)

        # move the time window of timeIndex and waveform every 5 seconds
        if int(time.time()*100) - nowtime.value >= 500:
            waveform_buffer_start_time.value += 500
            waveform_buffer[:, 0:5500] = waveform_buffer[:, 500:6000]
            
            # the updated waveform is fill in with 0
            waveform_buffer[:, 5500:6000] = torch.zeros((waveform_buffer.shape[0],500))
            nowtime.value += 500        

# picking: pick and send pick_msg to PICK_RING
def MultiStationWarning_NO_Picker(waveform_buffer, key_index,env_config, key_cnt, 
                        stationInfo, device,target_city,target_city_plot,
                        logfilename_warning,logfilename_notify,upload_TF,
                        warning_plot_TF):
    
    MyModule = PyEW.EWModule(int(env_config["WAVE_RING_ID"]), int(env_config["PYEW_MODULE_ID"]), int(env_config["PYEW_INST_ID"]), 30.0, False)
    MyModule.add_ring(int(env_config["WAVE_RING_ID"])) # WAVE_RING
    MyModule.add_ring(int(env_config["PICK_RING_ID"])) # PICK_RING
    # MyModule.add_ring(int(env_config["OUTPUT_RING_ID"])) # OUTPUT_RING
    
    stations_table = json.load(open(env_config["Multi_Station_Table_FILEPATH"], 'r'))
    stations_table_coords = {}
    dataDepth = {}
    for key in stations_table.keys():
        key_sub = f"{key.split(',')[0]},{key.split(',')[1]}"
        stations_table_coords[key_sub] = stations_table[key]
        dataDepth[key_sub] = key.split(',')[2]
        
        
    #預測要的東西
    pga_thresholds = np.array(env_config["MULTISTATION_THRESHOLDS"].split(','),dtype=np.float32)
    alpha = np.array(env_config["MULTISTATION_ALPHAS"].split(','),dtype=np.float32)
    model_path = env_config["Multi_Station_CHECKPOINT_FILEPATH"]
    config = json.load(open(os.path.join(env_config["Multi_Station_Config_PATH"]), 'r'))
    model_params = config['model_params'].copy()
    ens_id = 6
    model_params['rotation'] = np.pi / ens_id * 4 / (10 - 1)
    model = models.build_transformer_model(**config['model_params'], device=device, trace_length=3000).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device)['model_weights'], strict=False)

    model.eval()

    channel_tail = ['Z', 'N', 'E']   

    # 記錄目前 year, month, day，用於刪除過舊的 log files
    cur = datetime.fromtimestamp(time.time())
    system_year, system_month, system_day = cur.year, cur.month, cur.day

    # # sleep 120 seconds, 讓波型先充滿 noise，而不是 0
    # if env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'Palert':
    #     print('pending...')
    #     # time.sleep(120)
    
    cnt=0
    while True:
        cur_waveform_buffer, cur_key_index = waveform_buffer.clone(), key_index.copy()
        
        
        # skip if there is no waveform in buffer or key_index is collect faster than waveform
        if int(key_cnt.value) == 0 or key_cnt.value < len(key_index):
            continue
       
        # collect the indices of stations that contain 3-channel waveforms
        toPredict_idx, VtoA_idx, toPredict_scnl = [], [], []
        for k, v in cur_key_index.items():
            tmp = k.split('_')
            if env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'CWB':
                if tmp[1][-1] == 'Z':             
                    try:
                        tmp_idx = [cur_key_index[f"{tmp[0]}_{tmp[1][:2]}{i}_{tmp[2]}_{tmp[3]}"] for i in channel_tail]    # for channel = XXZ
                        toPredict_idx.append(tmp_idx)
                    
                        toPredict_scnl.append(f"{tmp[0]}_{tmp[1]}_{tmp[2]}_{tmp[3]}")

                        # collect indices where measurement is velocity    
                        if tmp[1][:2] in ['HH', 'EH']:     
                            VtoA_idx.append(True)
                        else:
                            VtoA_idx.append(False)

                    except Exception as e:
                        continue
            else:
                if tmp[1][-1] in ['1', '4', '7']:
                    try:
                        tmp_idx = [cur_key_index[f"{tmp[0]}_{tmp[1][:-1]}{i}_{tmp[2]}_{tmp[3]}"] for i in range(int(tmp[1][-1]), int(tmp[1][-1])+3)]    # for Ch1-9
                        toPredict_idx.append(tmp_idx)
                    
                        toPredict_scnl.append(f"{tmp[0]}_{tmp[1]}_{tmp[2]}_{tmp[3]}")

                        # collect indices where measurement is velocity    
                        if tmp[1][-1] in ['4', '7']:       
                            VtoA_idx.append(True)
                        else:
                            VtoA_idx.append(False)

                    except Exception as e:
                        continue
        
        # skip if there is no station need to predict
        
        if len(toPredict_idx) == 0:
            continue

        # take only 3000-sample waveform
        toPredict_wave = cur_waveform_buffer[torch.tensor(toPredict_idx, dtype=torch.long)][:, :, :3000]
        
        toPredict_scnl = np.array(toPredict_scnl)
        VtoA_idx = np.array(VtoA_idx)
       
        # get the factor and coordinates of stations
        if env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'CWB':
            station_factor_coords = get_Palert_CWB_coord(toPredict_scnl, stationInfo)

            # count to gal
            factor = np.array([f[-1] for f in station_factor_coords]).astype(float)
            toPredict_wave = toPredict_wave/factor[:, None, None]
        else:
            station_factor_coords = get_coord_factor(toPredict_scnl, stationInfo)

            # multiply with factor to convert count to 物理量
            factor = np.array([f[-1] for f in station_factor_coords])
            toPredict_wave = toPredict_wave*factor[:, :, None]
        
        
        # preprocess
        # 1) convert traces to acceleration
        # 2) 1-45Hz bandpass filter
        # 3) Z-score normalization
        # 4) calculate features: Characteristic, STA, LTA
        # toPredict_wave = filter(toPredict_wave)
        toPredict_wave = v_to_a(toPredict_wave, VtoA_idx)
        # toPredict_wave = z_score(toPredict_wave)
        # toPredict_wave = calc_feats(toPredict_wave)
        
        waveforms = np.zeros((1, len(stations_table),3,3000))
        metadata = np.zeros((1, len(stations_table) ,3))
        position_list = {}
        for i  in range(len(station_factor_coords)):
            station = station_factor_coords[i]
            try:
                depth = dataDepth[f"{station[1]},{station[0]}"]
                station_key = f"{station[1]},{station[0]},{depth}"
                postion = stations_table[station_key]
                position_list[postion] = True
                waveforms[0,postion] = toPredict_wave[i]
                metadata[0,postion] = np.array([station[1],station[0],depth])
            except:
                continue
        
        #(1,250,3000,3)
        waveforms = np.transpose(waveforms,(0,1,3,2))    
        metadata = location_transformation(metadata)
        

        
        mask = np.logical_and((metadata == 0).all(axis=(1, 2)), (waveforms == 0).all(axis=(1, 2, 3)))
        waveforms[mask, 0, 0, 0] = 1e-9
        metadata[mask, 0, 0] = 1e-9
        
        
        #  predict
        toPredict_wave = torch.FloatTensor(waveforms).to(device)/100
        toPredict_coords = torch.FloatTensor(metadata).to(device)
        
        # for i in range(250):
        #     if(i not in position_list.keys()):
        #         continue
        #     print(i)
        #     plt.plot(toPredict_wave[0,i,:,0])
        #     plt.savefig(f'{i}.png')
        #     plt.clf()
        #     plt.close()
            
        with torch.no_grad():
            pga_pred = model(toPredict_wave,toPredict_coords).cpu()

        pga_times_pre = np.zeros((pga_thresholds.shape[0],pga_pred.shape[1]), dtype=int)
            
        for j,log_level in enumerate(np.log10(pga_thresholds * 9.81)):
            prob = torch.sum(
                pga_pred[:, :, :, 0] * (1 - norm.cdf((log_level - pga_pred[:, :, :, 1]) / pga_pred[:, :, :, 2])),
                axis=-1)
            exceedance = np.squeeze(prob > alpha[j])  # Shape: stations, 1
            pga_times_pre[j] = exceedance
            
        #(250,4)
        pga_times_pre = np.transpose(pga_times_pre,(1,0))   
        
        #檢查是否有新要預警的測站
        warn_Flag = False
        warning_msg=""
        for city_index in range(pga_times_pre.shape[0]):
            #update 表格
            if(set(target_city[city_index][-1]) != set(pga_times_pre[city_index])):
                indices = [index for (index, item) in enumerate(pga_times_pre[city_index]) if item ==1 ]
                for warning_thresholds in range(len(pga_thresholds)):
                    if(warning_thresholds in indices):
                        target_city[city_index][-1][warning_thresholds] += 1
                    else:
                        target_city[city_index][-1][warning_thresholds] += 0
                    
                if (len(indices)!=0):
                    Flag = True
                    for indice in indices:
                        for index in range(indice,-1,-1):
                            if(target_city[city_index][-1][index]==0):
                                #不預警
                                Flag=False
                        if (not Flag) :
                            target_city[city_index][-1][indice] -= 1
                        if(Flag) and (target_city[city_index][-1][indice]>1):
                            Flag = False
                            
                    if Flag:
                        print(f"Warning time: {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:{target_city[city_index][0]},{target_city[city_index][-1]}\n")
                        warning_msg += f"{cnt} Warning time: {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:"
                        warning_msg += f"{target_city[city_index][0]},三級:{target_city[city_index][-1][0]},四級:{target_city[city_index][-1][1]},五弱級:{target_city[city_index][-1][2]},五強級:{target_city[city_index][-1][3]}\n"
                        target_city_plot.append(target_city)
                        warn_Flag = True
                        cnt += 1
    
        if warn_Flag:
            warning_logfile = f"./warning_log/log/{cur.year}-{cur.month}-{cur.day}_warning.log"
            # multi_station_msg_notify(warning_msg)
            # 已經是系統時間的隔天，檢查有沒有過舊的 log file，有的話將其刪除
            if f"{system_year}-{system_month}-{system_day}" != f"{cur.year}-{cur.month}-{cur.day}" or True:
                toDelete_picking = cur - timedelta(days=int(env_config['DELETE_PICKINGLOG_DAY']))

                toDelete_picking_filename = f"./warning_log/log/{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_warning.log"
                if os.path.exists(toDelete_picking_filename):
                    os.remove(toDelete_picking_filename)
                    
                logfilename_warning.value = f"./warning_log/log/{system_year}-{system_month}-{system_day}_warning.log"
                logfilename_notify.value = glob.glob("./warning_log/notify/*")
                upload_TF.value += 1
                
                #reset system time
                system_year, system_month, system_day = cur.year, cur.month, cur.day
                
            # writing picking log file
            with open(warning_logfile,"a") as pif:
                pif.write(warning_msg)
                pif.write('\n')
                pif.close()           
            warning_plot_TF.value += 1    

def MultiStationWarning(waveform_buffer, key_index,env_config, key_cnt, 
                        stationInfo, device,target_city,target_city_plot,
                        logfilename_warning,logfilename_notify,upload_TF,
                        warning_plot_TF,waveform_buffer_start_time):
    
    

    MyModule = PyEW.EWModule(int(env_config["WAVE_RING_ID"]), int(env_config["PYEW_MODULE_ID"]), int(env_config["PYEW_INST_ID"]), 30.0, False)
    MyModule.add_ring(int(env_config["WAVE_RING_ID"])) # WAVE_RING
    MyModule.add_ring(int(env_config["PICK_RING_ID"])) # PICK_RING
    # MyModule.add_ring(int(env_config["OUTPUT_RING_ID"])) # OUTPUT_RING

    wave_count = 0
    # flush PICK_RING
    while MyModule.get_bytes(1, int(env_config["PICK_MSG_TYPE"])) != (0, 0):
        wave_count += 1
        continue
    
    stations_table = json.load(open(env_config["Multi_Station_Table_FILEPATH"], 'r'))
    dataDepth = {}
    stations_table_coords = {}
    for key in stations_table.keys():
        key_sub = f"{key.split(',')[0]},{key.split(',')[1]}"
        stations_table_coords[key_sub] = stations_table[key]
        dataDepth[key_sub] = key.split(',')[2]   
        
    #預測要的東西
    pga_thresholds = np.array(env_config["MULTISTATION_THRESHOLDS"].split(','),dtype=np.float32)
    alpha = np.array(env_config["MULTISTATION_ALPHAS"].split(','),dtype=np.float32)
    model_path = env_config["Multi_Station_CHECKPOINT_FILEPATH"]
    config = json.load(open(os.path.join(env_config["Multi_Station_Config_PATH"]), 'r'))
    model_params = config['model_params'].copy()
    ens_id = 6
    model_params['rotation'] = np.pi / ens_id * 4 / (10 - 1)
    model = models.build_transformer_model(**config['model_params'], device=device, trace_length=3000).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device)['model_weights'], strict=False)

    model.eval()


    # 記錄目前 year, month, day，用於刪除過舊的 log files
    cur = datetime.fromtimestamp(time.time())
    system_year, system_month, system_day = cur.year, cur.month, cur.day

    # # sleep 120 seconds, 讓波型先充滿 noise，而不是 0
    # if env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'Palert':
    #     print('pending...')
    #     # time.sleep(120)
    # a waiting list for model to process if multiple stations comes into PICK_RING simultaneously
    wait_list = []
    
    
    # listen PICK_RING
    Test = False
    cnt = 0
    # collect the indices of stations that contain 3-channel waveforms
    position_list = {}
    while True:
        
        # get picked station in PICK_RING
        pick_msg = MyModule.get_bytes(1, int(env_config["PICK_MSG_TYPE"]))
        
        # if there's no data and waiting list is empty
        if pick_msg == (0, 0) and len(wait_list) == 0:
            continue

        # if get picked station, then get its info and add to waiting list
        if pick_msg != (0, 0):
            pick_str = pick_msg[2][:pick_msg[1]].decode("utf-8")

            # the quality of wave should not greater than 2
            if int(pick_str[-5]) > 2:
                continue

            # the using time period for picking module should be 3 seconds
            if int(pick_str[-1]) !=3:
                continue
            wait_list.append(pick_str)

        #append data
        station_index = 0
        waveforms = np.zeros((1, len(stations_table),3,3000))
        metadata = np.zeros((1, len(stations_table) ,3))
        for pick_info in wait_list:
            
            pick_info = pick_info.split(' ')
            # print(f"pick_info:{datetime.fromtimestamp(float(pick_info[10]))}")
            pick_index = int((float(time.time())-float(pick_info[10]))*100)
            # print(f"pick_index:{pick_index}")
            station = pick_info[0]
            channel = pick_info[1]
            network = pick_info[2]
            location = pick_info[3]
            scnl = f"{station}_{channel}_{network}_{location}"
            station_coord_factor = get_coord_factor(np.array([scnl]), stationInfo)[0]
            
            
            if env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'TSMIP':
                scnl_z = f"{station}_{channel[:-1]}Z_{network}_{location}"
                scnl_n = f"{station}_{channel[:-1]}N_{network}_{location}"
                scnl_e = f"{station}_{channel[:-1]}E_{network}_{location}"
            else:
                # for tankplayer testing
                if channel == 'HHZ': 
                    channel = ['Ch7', 'Ch8', 'Ch9']
                elif channel == 'EHZ': 
                    channel = ['Ch4', 'Ch5', 'Ch6']
                elif channel == 'HLZ': 
                    channel = ['Ch1', 'Ch2', 'Ch3']
                    
                scnl_z = f"{station}_{channel[0]}_{network}_{location}"
                scnl_n = f"{station}_{channel[1]}_{network}_{location}"
                scnl_e = f"{station}_{channel[2]}_{network}_{location}"
            
            # One of 3 channel is not in key_index(i.e. waveform)
            if (scnl_z not in key_index) or (scnl_n not in key_index) or (scnl_e not in key_index):
                print(f"{scnl} one of 3 channel missing")
                wait_list.pop(station_index)
                if(len(wait_list)>0):
                    continue
                else:
                  break
            
            # cannot search station Info
            if (station_coord_factor[1]==-1):
                print(f"{scnl} not find in stationInfo")
                wait_list.pop(station_index)
                if(len(wait_list)>0):
                    continue
                else:
                  break
              
              
            factor = np.array(station_coord_factor[-1])
            time_before = 0
            # get waveform of z,n,e starting from ptime
            start_index = max(3000-pick_index-time_before,0)
            
            hlz = waveform_buffer[key_index[scnl_z]][start_index:3000]*0.01
            hln = waveform_buffer[key_index[scnl_n]][start_index:3000]*0.01
            hle = waveform_buffer[key_index[scnl_e]][start_index:3000]*0.01
        
            inp = torch.cat((hlz.unsqueeze(0), hln.unsqueeze(0), hle.unsqueeze(0)), dim=0)  # missing factor
            inp = inp*factor[:,None]
            
            
            depth = dataDepth[f"{station_coord_factor[1]},{station_coord_factor[0]}"]
            station_key = f"{station_coord_factor[1]},{station_coord_factor[0]},{depth}"
            position = stations_table[station_key]
            position_list[position] = True
            waveforms[0,position,:,0:inp.shape[1]] = inp
            waveforms = waveforms - np.mean(waveforms, axis=-1, keepdims=True)
            
            # plt.plot(waveform_buffer[key_index[scnl_z]][0:3000]*0.01)
            # plt.savefig(f'./img/{cnt}_origin.png')
            # plt.clf()
            # plt.close() 
            
            # if(position==236):
            #     print(f"start_index:{start_index}")
            # plt.plot(waveforms[0,236,0,:])
            # plt.savefig(f'./img/{cnt}_{236}.png')
            # plt.clf()
            # plt.close() 
            # cnt+=1
            metadata[0,position] = np.array([station_coord_factor[1],station_coord_factor[0],depth])
            
            station_index+=1
            
        # #紀錄地震開始第一個測站Picking時間
        # if (len(toPredict_scnl)==1):
        #     # get the filenames
        #     cur = datetime.fromtimestamp(time.time())
        #     warning_logfile = f"./warning_log/warning/{cur.year}-{cur.month}-{cur.day}_warning.log"
        #     start_count = datetime.utcfromtimestamp(time.time())
        #     with open(warning_logfile,"a") as pif:
        #         pif.write(f"First Station Picking time: {start_count.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        #         pif.write('='*25)
        #         pif.write('\n')
            
        
        # # 每小時發一個 notify，證明系統還活著
        # # if f"{system_year}-{system_month}-{system_day}-{system_hour}" != f"{cur.year}-{cur.month}-{cur.day}-{cur.hour}":
        # #     multi_station_msg_notify("system live")
        # #     system_hour = cur.hour
        
        #(1,250,3000,3)
        input_waveforms = np.transpose(waveforms,(0,1,3,2))    
        input_metadata = location_transformation(metadata)
        input_waveforms = torch.Tensor(input_waveforms)
        input_metadata = torch.Tensor(input_metadata)
        
        
        with torch.no_grad():
            pga_pred = model(input_waveforms,input_metadata).cpu()

        pga_times_pre = np.zeros((pga_thresholds.shape[0],pga_pred.shape[1]), dtype=int)
            
        for j,log_level in enumerate(np.log10(pga_thresholds * 9.81)):
            prob = torch.sum(
                pga_pred[:, :, :, 0] * (1 - norm.cdf((log_level - pga_pred[:, :, :, 1]) / pga_pred[:, :, :, 2])),
                axis=-1)
            exceedance = np.squeeze(prob > alpha[j])  # Shape: stations, 1
            pga_times_pre[j] = exceedance
            
        #(250,4)
        pga_times_pre = np.transpose(pga_times_pre,(1,0))   
        
        #檢查是否有新要預警的測站
        warn_Flag = False
        warning_msg=""
        for city_index in range(pga_times_pre.shape[0]):
            #update 表格
            if(set(target_city[city_index][-1]) != set(pga_times_pre[city_index])):
                indices = [index for (index, item) in enumerate(pga_times_pre[city_index]) if item ==1 ]
                for warning_thresholds in range(len(pga_thresholds)):
                    if(warning_thresholds in indices):
                        target_city[city_index][-1][warning_thresholds] += 1
                    else:
                        target_city[city_index][-1][warning_thresholds] += 0
                    
                if (len(indices)!=0):
                    Flag = True
                    for indice in indices:
                        for index in range(indice,-1,-1):
                            if(target_city[city_index][-1][index]==0):
                                #不預警
                                Flag=False
                        if (not Flag) :
                            target_city[city_index][-1][indice] -= 1
                        if(Flag) and (target_city[city_index][-1][indice]>1):
                            Flag = False
                            
                    if Flag:
                        print(f"Warning time: {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:{target_city[city_index][0]},{target_city[city_index][-1]}\n")
                        warning_msg += f"{cnt} Warning time: {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:"
                        warning_msg += f"{target_city[city_index][0]},三級:{target_city[city_index][-1][0]},四級:{target_city[city_index][-1][1]},五弱級:{target_city[city_index][-1][2]},五強級:{target_city[city_index][-1][3]},,六弱級:{target_city[city_index][-1][4]}\n"
                        target_city_plot.append(target_city)
                        warn_Flag = True
                        cnt += 1
    
        if warn_Flag:
            warning_logfile = f"./warning_log/log/{cur.year}-{cur.month}-{cur.day}_warning.log"
            # multi_station_msg_notify(warning_msg)
            # 已經是系統時間的隔天，檢查有沒有過舊的 log file，有的話將其刪除
            if f"{system_year}-{system_month}-{system_day}" != f"{cur.year}-{cur.month}-{cur.day}" or True:
                toDelete_picking = cur - timedelta(days=int(env_config['DELETE_PICKINGLOG_DAY']))

                toDelete_picking_filename = f"./warning_log/log/{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_warning.log"
                if os.path.exists(toDelete_picking_filename):
                    os.remove(toDelete_picking_filename)
                    
                logfilename_warning.value = f"./warning_log/log/{system_year}-{system_month}-{system_day}_warning.log"
                logfilename_notify.value = glob.glob("./warning_log/notify/*")
                upload_TF.value += 1
                
                #reset system time
                system_year, system_month, system_day = cur.year, cur.month, cur.day
                
            # writing picking log file
            with open(warning_logfile,"a") as pif:
                pif.write(warning_msg)
                pif.write('\n')
                pif.close()           
            warning_plot_TF.value += 1    


# multi-station prediction
# picking: pick and send pick_msg to PICK_RING
def PickHandlerMultiStation_TEAMS(needed_wave,waveform_buffer, key_index, nowtime, waveform_buffer_start_time, 
                            env_config, target_city,warning_plot_TF,stationInfo,target_city_plot,needed_wave_input_plot,
                            logfilename_warning,logfilename_notify,upload_TF):
    
    

    MyModule = PyEW.EWModule(int(env_config["WAVE_RING_ID"]), int(env_config["PYEW_MODULE_ID"]), int(env_config["PYEW_INST_ID"]), 30.0, False)
    MyModule.add_ring(int(env_config["WAVE_RING_ID"])) # WAVE_RING
    MyModule.add_ring(int(env_config["PICK_RING_ID"])) # PICK_RING
    # MyModule.add_ring(int(env_config["OUTPUT_RING_ID"])) # OUTPUT_RING

    wave_count = 0
    # flush PICK_RING
    while MyModule.get_bytes(1, int(env_config["PICK_MSG_TYPE"])) != (0, 0):
        wave_count += 1

        continue
    
    # for _ in tqdm(range(30)):
    #   time.sleep(1)
 
    # initialize and load model
    config = json.load(open(os.path.join(env_config["MULTISTATION_FILEPATH"]), 'r'))
    _, model = build_transformer_model(**config['model_params'],trace_length=3000,config=config)
    model.build([(None, 25, 3000, 3),(None, 25, 3),(None, 15, 3)])
    model.load_weights(env_config["MULTISTATION_CHECKPOINT_FILEPATH"])

    # a waiting list for model to process if multiple stations comes into PICK_RING simultaneously
    wait_list = deque()
    needed_station = []
    needed_coord = np.zeros((1,25, 3))
    needed_wave_tensor = np.zeros((1,25,3,3000))
    
    # 記錄目前 year, month, day，用於刪除過舊的 log files
    cur = datetime.fromtimestamp(time.time())
    system_year, system_month, system_day, system_hour = cur.year, cur.month, cur.day,cur.hour
    
    # read .dat to a list of lists
    dataDepth = {}
    datContent = [i.strip().split() for i in open("./nsta/nsta24.dat").readlines()]
    target_coord = pd.read_csv(env_config["MULTISTATION_TARTGET_COORD"])
    target_coord_input = []
    for i in range(len(target_coord)):
        target_coord_input.append([target_coord.iloc[i]['lon'],target_coord.iloc[i]['lat'],target_coord.iloc[i]['depth']])
    target_coord_input = np.array(target_coord_input).reshape(-1,15,3)
    
    # make depth table
    for row in datContent:
        key = f"{row[0]}_HLZ_{row[7]}_0{row[5]}"
        dataDepth[key] = float(row[3])
        
    #預測要的東西
    pga_thresholds = np.array(env_config["MULTISTATION_THRESHOLDS"].split(','),dtype=np.float32)
    alpha = np.array(env_config["MULTISTATION_ALPHAS"].split(','),dtype=np.float32)
            
            
    PICK_MSG_TYPE = int(env_config["PICK_MSG_TYPE"])
    # listen PICK_RING
    Test = False
    cnt = 0
    
    while True:
        
        log_msg = "============================"
        if Test==False:
            # get picked station in PICK_RING
            pick_msg = MyModule.get_bytes(1, PICK_MSG_TYPE)
            
            # if there's no data and waiting list is empty
            if pick_msg == (0, 0) and len(wait_list) == 0 and len(needed_station)==0:
                continue

            # if get picked station, then get its info and add to waiting list
            if pick_msg != (0, 0):
                pick_str = pick_msg[2][:pick_msg[1]].decode("utf-8")

                # the quality of wave should not greater than 2
                if int(pick_str[-5]) > 2:
                    continue

                # the using time period for picking module should be 3 seconds
                if int(pick_str[-1]) !=3:
                    continue
                wait_list.append(pick_str)

            #append first 25 station
            while (len(wait_list)!=0) and (len(needed_station)<25):
               #print(f"get data Start:{datetime.utcfromtimestamp(time.time())}")
                log_msg += "\n[" + str(time.time()) + "] " + str(wait_list[0])
                # get the first one data in waiting list
                pick_info = wait_list[0].split()
                station = pick_info[0]
                channel = pick_info[1]
                network = pick_info[2]
                location = pick_info[3]
                scnl = f"{station}_{channel}_{network}_{location}"
                
                scnl_z = f"{station}_{channel[:-1]}Z_{network}_{location}"
                scnl_n = f"{station}_{channel[:-1]}N_{network}_{location}"
                scnl_e = f"{station}_{channel[:-1]}E_{network}_{location}"

                # for tankplayer testing
                # if channel == 'HHZ': 
                #    channel = ['Ch7', 'Ch8', 'Ch9']
                # elif channel == 'EHZ': 
                #    channel = ['Ch4', 'Ch5', 'Ch6']
                # elif channel == 'HLZ': 
                #    channel = ['Ch1', 'Ch2', 'Ch3']
                    
                # scnl_z = f"{station}_{channel[0]}_{network}_{location}"
                # scnl_n = f"{station}_{channel[1]}_{network}_{location}"
                # scnl_e = f"{station}_{channel[2]}_{network}_{location}"
                # for tankplayer testing
                

                # One of 3 channel is not in key_index(i.e. waveform)
           
                if (scnl_z not in key_index) or (scnl_n not in key_index) or (scnl_e not in key_index):
                    print(f"{scnl} one of 3 channel missing")
                    wait_list.popleft()
                    continue

                needed_station.append(scnl)
                wait_list.popleft()
            
            #紀錄地震開始第一個測站Picking時間
            if (len(needed_station)==1):
                # get the filenames
                cur = datetime.fromtimestamp(time.time())
                warning_logfile = f"./warning_log/warning/{cur.year}-{cur.month}-{cur.day}_warning.log"
                start_count = datetime.utcfromtimestamp(time.time())
                with open(warning_logfile,"a") as pif:
                    pif.write(f"First Station Picking time: {start_count.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    pif.write('='*25)
                    pif.write('\n')
                
            #清空前25測站資訊
            if(len(needed_station)>=1):    
                now_sub = (datetime.utcfromtimestamp(time.time()) -  start_count)
                if(now_sub.total_seconds()>30):
                    needed_station = []
                    with open(warning_logfile,"a") as pif:
                        pif.write(f"clear first 25 station:{datetime.utcfromtimestamp(time.time())}")
                        pif.write('='*25)
                        pif.write('\n')
                    #multi_station_msg_notify("第一個測站pick後60s清空測站")
                    for i in range(len(target_coord)):
                        target_city[i] = [f"{target_coord.iloc[i]['lon']}_{target_coord.iloc[i]['lat']}",target_coord.iloc[i]['lon'],target_coord.iloc[i]['lat'],[0,0,0,0]]
                    needed_coord = np.zeros((1,25, 3))
                    needed_wave_tensor = np.zeros((1,25,3,3000))
            
            # 每小時發一個 notify，證明系統還活著
            # if f"{system_year}-{system_month}-{system_day}-{system_hour}" != f"{cur.year}-{cur.month}-{cur.day}-{cur.hour}":
            #     multi_station_msg_notify("system live")
            #     system_hour = cur.hour


            # append first 25 station waveform
            for station_index in range(len(needed_station)):
                scnl = needed_station[station_index]
                # scnl = f"{station}_{channel}_{network}_{location}"
                station = scnl.split('_')[0]
                channel = scnl.split('_')[1]
                network = scnl.split('_')[2]
                location = scnl.split('_')[3]
                
                # search depth
                if(scnl in dataDepth.keys()):
                    coords = np.array([pick_info[5],pick_info[4],dataDepth[scnl]/100])
                else:
                    coords = np.array([pick_info[5],pick_info[4],0.0])
                    
                needed_coord[0,station_index,:] = coords
    
                scnl_z = f"{station}_{channel[:-1]}Z_{network}_{location}"
                scnl_n = f"{station}_{channel[:-1]}N_{network}_{location}"
                scnl_e = f"{station}_{channel[:-1]}E_{network}_{location}"
                
                # for tankplayer testing
                # if channel == 'HHZ': 
                #    channel = ['Ch7', 'Ch8', 'Ch9']
                # elif channel == 'EHZ': 
                #    channel = ['Ch4', 'Ch5', 'Ch6']
                # elif channel == 'HLZ': 
                #    channel = ['Ch1', 'Ch2', 'Ch3']
                    
                # scnl_z = f"{station}_{channel[0]}_{network}_{location}"
                # scnl_n = f"{station}_{channel[1]}_{network}_{location}"
                # scnl_e = f"{station}_{channel[2]}_{network}_{location}"
                # for tankplayer testing
                
                # get the index of z,n,e in wavoform
                z_waveform_index = key_index[scnl_z]
                n_waveform_index = key_index[scnl_n]
                e_waveform_index = key_index[scnl_e]


                # get waveform of z,n,e starting from ptime
                hlz = waveform_buffer[z_waveform_index][0:3000]
                hln = waveform_buffer[n_waveform_index][0:3000]
                hle = waveform_buffer[e_waveform_index][0:3000]
                
                inp = torch.cat((hlz.unsqueeze(0), hln.unsqueeze(0), hle.unsqueeze(0)), dim=0)  # missing factor
                
                # plt.plot(inp.T)
                # plt.savefig(f"./plott/{cnt}.png")
                # plt.clf()
                # cnt+=1
                #(1,25,3,3000)
                needed_wave_tensor[0,station_index] = inp

            #處理factor
            if(len(needed_station)>1):
                try:
                    station_factor_coords = get_Palert_CWB_coord(np.array(needed_station), stationInfo)
                    # count to gal
                    factor = np.array([f[-1] for f in station_factor_coords]).astype(float)
                    # print(factor)
                    needed_wave_tensor[:,:len(needed_station)] = needed_wave_tensor[:,:len(needed_station)]/factor[:, None, None]
                except:
                    station_factor_coords = get_coord_factor(np.array(needed_station), stationInfo)
                    # count to gal
                    factor = np.array([f[-1] for f in station_factor_coords]).astype(float)
                    needed_wave_tensor[:,:len(needed_station)] = needed_wave_tensor[:,:len(needed_station)]/factor[None,:, :,None]
        else:
            pass
        #print(f"get data End:{datetime.utcfromtimestamp(time.time())}")
        
        # print(f"{cnt}.png")
        # print(needed_wave_tensor.shape)
        # plot_wave(needed_wave_tensor,f"{cnt}.png")
        # cnt+=1
        
        if(len(needed_coord)>0):
            #波型、座標、目標座標
            needed_wave_tensor_input = np.transpose(needed_wave_tensor,(0,1,3,2))/100
            needed_wave_tensor_input = np.repeat(needed_wave_tensor_input,target_coord_input.shape[0], axis=0)
            needed_coord_input = np.tile(needed_coord,target_coord_input.shape[0]).reshape(target_coord_input.shape[0],25,3)
            needed_coord_input = location_transformation(needed_coord_input)
            target_coord_input = location_transformation(target_coord_input)
            
            # print(f"needed_coord_input:{needed_coord_input}")
            # print(f"target_coord_input:{target_coord_input}")
            # print(f"{cnt}.png")
            # print(needed_wave_tensor.shape)
            # plot_wave(needed_wave_tensor[0],f"{cnt}.png")
            # cnt+=1
            pga_pred = model.predict([needed_wave_tensor_input,needed_coord_input,target_coord_input],verbose = 0)['pga']
            log_msg += "\n[" + str(time.time()) + "] end predict"
            pga_times_pre = np.zeros((pga_thresholds.shape[0],target_coord_input.shape[0],pga_pred.shape[1]), dtype=int)
            for j,log_level in enumerate(np.log10(pga_thresholds * 9.81)):
                prob = np.sum(
                    pga_pred[:, :, :, 0] * (1 - norm.cdf((log_level - pga_pred[:, :, :, 1]) / pga_pred[:, :, :, 2])),
                    axis=-1)
                exceedance = np.squeeze(prob > alpha[j])  # Shape: stations, 1
                pga_times_pre[j] = exceedance
                
            # output:(45,4)
            pga_times_pre = pga_times_pre.reshape(-1,pga_thresholds.shape[0])
            
            #檢查是否有新要預警的測站
            warn_Flag = False
            warning_msg=""
            for city_index in range(pga_times_pre.shape[0]):
                #update 表格
                if(set(target_city[city_index][-1]) != set(pga_times_pre[city_index])):
                    indices = [index for (index, item) in enumerate(pga_times_pre[city_index]) if item ==1 ]
                    for warning_thresholds in range(len(pga_thresholds)):
                        if(warning_thresholds in indices):
                            target_city[city_index][-1][warning_thresholds] += 1
                        else:
                            target_city[city_index][-1][warning_thresholds] += 0
                       
                    if (len(indices)!=0):
                        Flag = True
                        for indice in indices:
                            for index in range(indice,-1,-1):
                                if(target_city[city_index][-1][index]==0):
                                    #不預警
                                    Flag=False
                            if (not Flag) :
                                target_city[city_index][-1][indice] -= 1
                            if(Flag) and (target_city[city_index][-1][indice]>1):
                                Flag = False
                                
                        if Flag:
                            print(f"Warning time: {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:{target_city[city_index][0]},{target_city[city_index][-1]}\n")
                            warning_msg += f"{cnt} Warning time: {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:"
                            warning_msg += f"{target_city[city_index][0]},三級:{target_city[city_index][-1][0]},四級:{target_city[city_index][-1][1]},五弱級:{target_city[city_index][-1][2]},五強級:{target_city[city_index][-1][3]}\n"
                            target_city_plot.append(target_city)
                            needed_wave_input_plot.append(needed_wave_tensor)
                            warn_Flag = True
                            cnt += 1
        
            if warn_Flag:
                
                # multi_station_msg_notify(warning_msg)
                # 已經是系統時間的隔天，檢查有沒有過舊的 log file，有的話將其刪除
                if f"{system_year}-{system_month}-{system_day}" != f"{cur.year}-{cur.month}-{cur.day}" or True:
                    toDelete_picking = cur - timedelta(days=int(env_config['DELETE_PICKINGLOG_DAY']))

                    toDelete_picking_filename = f"./warning_log/warning/{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_warning.log"
                    if os.path.exists(toDelete_picking_filename):
                        os.remove(toDelete_picking_filename)
                        
                    logfilename_warning.value = f"./warning_log/warning/{system_year}-{system_month}-{system_day}_warning.log"
                    logfilename_notify.value = glob.glob("./warning_log/notify/*")
                    upload_TF.value += 1
                    
                    #reset system time
                    system_year, system_month, system_day = cur.year, cur.month, cur.day
                    
                # writing picking log file
                with open(warning_logfile,"a") as pif:
                    pif.write(warning_msg)
                    pif.write('\n')
                    pif.close()           
                warning_plot_TF.value += 1    
      
# plotting
def WarningShower(target_city_plot,warning_plot_TF,needed_wave_input_plot):
    
    while True:
        isNotify = False

        # don't plot, keep pending ...
        if warning_plot_TF.value == 0.0:
            continue

        cur_time = datetime.utcfromtimestamp(time.time())
        plot_filename = f"{cur_time.year}-{cur_time.month}-{cur_time.day}_{cur_time.hour}:{cur_time.minute}:{cur_time.second}"
        
        if warning_plot_TF.value != 0.0:
            filename = f"./warning_log/notify/{plot_filename}.png"
            wave_filename = f"./warning_log/notify/{plot_filename}_wave.png"
            isNotify = True
        else:
            filename = f"./warning_log/notify/{plot_filename}.png"


        # send the prediction through line notify
        if isNotify:
            plot_wave(needed_wave_input_plot[-1],wave_filename)
            plot_taiwan(target_city_plot[-1],filename)
            multi_station_plot_notify(filename) 
            multi_station_plot_notify(wave_filename) 
            warning_plot_TF.value -= 1
  

if __name__ == '__main__':
    
    #basic code setting
    torch.multiprocessing.set_start_method('spawn')
    
    
    #folder create
    if not os.path.exists('./warning_log/log'):
        os.makedirs('./warning_log/log')
    if not os.path.exists('./warning_log/plot'):
        os.makedirs('./warning_log/plot')

    try:
        # create multiprocessing manager to maintain the shared variables
        manager = Manager()
        env_config = manager.dict()
        for k, v in dotenv_values(".env").items():
            env_config[k] = v
            
            
        # a deque from time-3000 to time for time index
        nowtime = Value('d', int(time.time()*100))
        waveform_buffer_start_time = Value('d', nowtime.value-3000)
        needed_wave = []
        # needed_wave_input = torch.zeros((3,25,3000,3),dtype=torch.float32).share_memory_()
        
        # a counter for accumulating key's count
        key_cnt = Value('d', int(0))
        # a dict for checking scnl's index of waveform
        key_index = manager.dict()

        # parameter for uploader
        logfilename_warning = manager.Value(c_char_p, 'hello')
        logfilename_notify = manager.Value(c_char_p, 'hello')
        upload_TF = Value('d', int(0))
        
        # create update table
        stations_table = json.load(open(env_config["Multi_Station_Table_FILEPATH"], 'r'))
        target_city={}
        target_city_plot = manager.list()
        needed_wave_input_plot = manager.list()
        count = 0
        for key in stations_table.keys():
            target_coord = key.split(',')
            target_city[count] = [f"{target_coord[0]}_{target_coord[1]}",target_coord[0],target_coord[1],[0,0,0,0,0]]
            count += 1 
        
        # to save all raw wave form data, which is the numpy array, and the shape is (station numbur*channel, 3000)
        waveform_buffer = torch.empty((int(env_config["N_PREDICTION_STATION"])*3, int(env_config["STORE_LENGTH"]))).share_memory_()

        # waveform that should be plotted in Shower
        waveform_plot = torch.empty((1, 3, 3000)).share_memory_()
        waveform_plot_prediction = torch.empty((1, 3000)).share_memory_()
        waveform_plot_isNotify = Value('d', int(0))      # 0: no notify sended, 1: notify sended
        waveform_plot_TF = Value('d', int(0))            # 0: plot!, 1: don't plot
        waveform_plot_picktime = Value('d', int(0))      # picking time for plotting
        warning_plot_TF = Value('d', int(0))

        # device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        # get the station's info
        # get the station list of palert
        if env_config['SOURCE'] == 'Palert':
            stationInfo = get_PalertStationInfo(env_config['PALERT_FILEPATH'])
        elif env_config['SOURCE'] == 'CWB':
            stationInfo = get_CWBStationInfo(env_config['NSTA_FILEPATH'])
        elif env_config['SOURCE'] == 'TSMIP':
            stationInfo = get_TSMIPStationInfo(env_config['TSMIP_FILEPATH'])
        else:
            stationInfo = get_StationInfo(env_config['NSTA_FILEPATH'], (datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f'))
        
        wave_saver = Process(target=WaveSaver, args=(env_config, waveform_buffer, key_index, nowtime, waveform_buffer_start_time, key_cnt, stationInfo))
        wave_saver.start()
        
        
        waining = Process(target=MultiStationWarning, args=(waveform_buffer, key_index,env_config, key_cnt, 
                                                            stationInfo, device,target_city,target_city_plot,
                                                            logfilename_warning,logfilename_notify,upload_TF,
                                                            warning_plot_TF,waveform_buffer_start_time))
        waining.start()


        # wave_shower = Process(target=WarningShower, args=(target_city_plot,warning_plot_TF,needed_wave_input_plot))
        # wave_shower.start()

        
        # uploader = Process(target=Uploader, args=(logfilename_warning, logfilename_notify, upload_TF))
        # uploader.start()
        
        
        wave_saver.join()
        waining.join()
        # wave_shower.join()
        # uploader.join()

    except KeyboardInterrupt:
        wave_saver.terminate()
        wave_saver.join()

        waining.terminate()
        waining.join()
        
        
        # pick_handler.terminate()
        # pick_handler.join()

        # picker.terminate()
        # picker.join()

        # wave_shower.terminate()
        # wave_shower.join()