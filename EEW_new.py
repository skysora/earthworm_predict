import time
import numpy as np
import torch
import torch.nn.functional as F

import multiprocessing
from multiprocessing import Process, Manager, Array, Value, Queue
from itertools import compress
import threading
import torch.multiprocessing
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
from cala import *
from decimal import Decimal

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
    # cnt = 0
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
            waveform_buffer[int(key_cnt.value)] = torch.zeros((1, int(env_config["STORE_LENGTH"])))
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
            waveform_buffer[:, 0:int(env_config["STORE_LENGTH"])-500] = waveform_buffer[:, 500:int(env_config["STORE_LENGTH"])]
            
            # the updated waveform is fill in with 0
            waveform_buffer[:, int(env_config["STORE_LENGTH"])-500:int(env_config["STORE_LENGTH"])] = torch.zeros((waveform_buffer.shape[0],500))
            nowtime.value += 500     
            
            # print(waveform_buffer.shape)
            # plt.plot(waveform_buffer[0,:])
            # plt.savefig(f'./img/{cnt}.png')
            # plt.clf()
            # plt.close() 
            # cnt+=1

def MultiStationWarning(waveform_buffer, key_index,env_config,stationInfo,device,target_city,target_city_plot,
                        logfilename_warning,logfilename_notify,upload_TF,
                        warning_plot_TF,target_waveform_plot,log_name,wait_list_plot,now_time):
    
    MyModule = PyEW.EWModule(int(env_config["WAVE_RING_ID"]), int(env_config["PYEW_MODULE_ID"]), int(env_config["PYEW_INST_ID"]), 30.0, False)
    MyModule.add_ring(int(env_config["WAVE_RING_ID"])) # WAVE_RING
    MyModule.add_ring(int(env_config["PICK_RING_ID"])) # PICK_RING
    # MyModule.add_ring(int(env_config["OUTPUT_RING_ID"])) # OUTPUT_RING

    wave_count = 0
    # flush PICK_RING
    while MyModule.get_bytes(1, int(env_config["PICK_MSG_TYPE"])) != (0, 0):
        wave_count += 1
        continue
            
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

    Pick_Time_dict = {}
    # 記錄目前 year, month, day，用於刪除過舊的 log files
    cur = datetime.fromtimestamp(time.time())
    system_year, system_month, system_day = cur.year, cur.month, cur.day

    # sleep 120 seconds, 讓波型先充滿 noise，而不是 0
    # if env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'Palert':
    #     print('pending...')
    #     time.sleep(30)
    
    # a waiting list for model to process if multiple stations comes into PICK_RING simultaneously
    #============================================================參數設定============================================================
    wait_list = []
    cnt = 0
    First_Station_Flag = False
    dataDepth = {}
    stations_table_coords={}
    stations_table = json.load(open(env_config["Multi_Station_Table_Model_FILEPATH"], 'r'))
    stations_table_chinese = json.load(open(env_config["Multi_Station_Table_Chinese_FILEPATH"], 'r'))
    stations_table_model = json.load(open(env_config["Multi_Station_Table_Model_FILEPATH"], 'r'))
    metadata = np.zeros((1, len(stations_table_model) ,3))
    for key in stations_table.keys():
        position = stations_table[key]
        metadata[0,position] = np.array([key.split(',')[0],key.split(',')[1],key.split(',')[2]])
        key_sub = f"{key.split(',')[0]},{key.split(',')[1]}"
        stations_table_coords[key_sub] = stations_table[key]
        dataDepth[key_sub] = key.split(',')[2]    
    first_station_index = None
    lengh = 3000
    channel_number = 3
    seconds_stop = 60
    time_before = 5
    now_index = (int(env_config["STORE_LENGTH"])//2)
    wait_list_position_dict={}
    #============================================================參數設定============================================================
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
    
        #紀錄地震開始第一個測站Picking時間
        if(not First_Station_Flag):
            First_Station_Flag = True
            # get the filenames
            create_file_cur = datetime.fromtimestamp(time.time())
            warning_logfile = f"./warning_log/log/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}_warning.log"
            first_station_time = datetime.fromtimestamp(time.time())
            with open(warning_logfile,"a") as pif:
                pif.write(f"Description,Picking_Time,Warning_Time,Station_Id,County,Township,Station_Chinese_Name,8gal,25gal,81gal,140gal,250gal,Label_0.8gal,Label_2.5gal,Label_8gal,Label_25gal,Label_81gal,Label_140gal,Label_250gal")
                pif.write('\n')
                
            
        if(First_Station_Flag):
            if((datetime.fromtimestamp(time.time()) - first_station_time).seconds >= seconds_stop):
                print(f"reset")
                First_Station_Flag = False
                log_name.value = warning_logfile
                target_waveform_plot.append(waveform_buffer)
                wait_list_plot.append(wait_list)
                now_time.value = time.time()
                warning_plot_TF.value += 1
                wait_list=[]
                wait_list=[]
                # create update table
                # count = 0
                # print("Create Table")
                # for key in stations_table.keys():
                #     target_coord = key.split(',')
                #     position = stations_table[key]
                #     target_city[count] = [stations_table_chinese[f"{position}"],target_coord[0],target_coord[1],[0,0,0,0,0]]
                #     count += 1 
                # print("Finish Create Table")
                count = 0
                print("Create Table")
                for key in stations_table.keys():
                    target_coord = key.split(',')
                    key = "%.4f,%.4f" % (float(target_coord[0]), float(target_coord[1]))
                    target_city[count] = [stations_table_chinese[key],target_coord[0],target_coord[1],[0,0,0,0,0]]
                    count += 1 
                print("Finish Create Table")
        
        #============================================================計算pick順序============================================================
        station_index = 0
        now_time.value = datetime.fromtimestamp(time.time())
        ok_wait_list = []
        scnl_list_position=[]
        for pick_info in wait_list:

            pick_info = pick_info.split(' ')
            station = pick_info[0]
            channel = pick_info[1]
            network = pick_info[2]
            location = pick_info[3]
            quality = pick_info[-3]
            scnl = f"{station}_{channel}_{network}_{location}"
            station_coord_factor = get_coord_factor(np.array([scnl]), stationInfo)[0]
            #============================================================篩選資料============================================================
            #判斷pick品質
            if(int(quality) != 0 and int(quality) != 1):
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
            # 同個測站不要重複
            if(f"{station}_{channel}_{network}" not in scnl_list_position):
                scnl_list_position.append(f"{station}_{channel}_{network}")
            else:
                print(f"{scnl} is duplicate")
                wait_list.pop(station_index)
                if(len(wait_list)>0):
                    continue
                else:
                  break
            #============================================================篩選資料============================================================
            depth = dataDepth[f"{station_coord_factor[1]},{station_coord_factor[0]}"]
            station_key = f"{station_coord_factor[1]},{station_coord_factor[0]},{depth}"
            position = stations_table[station_key]
            pick_index = now_index - int(float(datetime.timestamp((now_time.value))-float(pick_info[10]))*100)
            #pick位置要選最早的
            if(station_key in stations_table.keys()):
                position = stations_table[station_key]
                pick_index = now_index - int(float(datetime.timestamp((now_time.value))-float(pick_info[10]))*100)
                #pick位置要選最早的
                if(position in wait_list_position_dict.keys()):
                    if(wait_list_position_dict[position]>pick_index):
                        wait_list_position_dict[position] = pick_index
                    else:
                        continue
                else:
                    wait_list_position_dict[position] = pick_index
                
                ok_wait_list.append(pick_info)
            else:
                print(f"{station_key} not in station_table")
                pass

            station_index+=1
        
        if(len(wait_list_position_dict.keys()) > 0): 
            wait_list_sort_index = np.argsort(np.array(wait_list_position_dict.values()))
            first_station_index = list(wait_list_position_dict.values())[wait_list_sort_index[0]]
            first_station_position = list(wait_list_position_dict.keys())[wait_list_sort_index[0]]
            first_station_time = now_time.value -  timedelta(seconds=(now_index - first_station_index)/100)
        
        #============================================================計算pick順序============================================================
        
            
        #append 資料到正確位置
        station_index = 0
        waveforms = np.zeros((1, len(stations_table_model),channel_number,lengh))
        scnl_list_position = []
        for pick_info in wait_list:
            
            pick_info = pick_info.split(' ')
            station = pick_info[0]
            channel = pick_info[1]
            network = pick_info[2]
            location = pick_info[3]
            scnl = f"{station}_{channel}_{network}_{location}"
            station_coord_factor = get_coord_factor(np.array([scnl]), stationInfo)[0]
            factor = torch.Tensor(station_coord_factor[-1])
            
            #============================================================篩選資料============================================================
            
            #確認channel命名
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
                continue
            
            #============================================================篩選資料============================================================


            #============================================================拿資料放入正確的位置============================================================
            # get waveform of z,n,e starting from ptime
            depth = dataDepth[f"{station_coord_factor[1]},{station_coord_factor[0]}"]
            station_key = f"{station_coord_factor[1]},{station_coord_factor[0]},{depth}"
            position = stations_table[station_key]
            Pick_Time_dict[position] = f"{datetime.fromtimestamp(float(pick_info[10]))}"
            
            
            start_index = max(first_station_index-time_before*100,now_index-lengh)
            hlz = waveform_buffer[key_index[scnl_z]][start_index:now_index]*0.01
            hln = waveform_buffer[key_index[scnl_n]][start_index:now_index]*0.01
            hle = waveform_buffer[key_index[scnl_e]][start_index:now_index]*0.01
        
            inp = torch.cat((hlz.unsqueeze(0), hln.unsqueeze(0), hle.unsqueeze(0)), dim=0)  # missing factor
            inp = inp*factor[:,None]
            inp = inp - torch.mean(inp, dim=-1, keepdims=True)
        
            waveforms[0,position,:,0:inp.shape[1]] = inp
            #============================================================拿資料放入正確的位置============================================================
            #============================================================確認輸入波型============================================================
            pga_threshold = [0.081,0.25,0.81,2.5,8.1,14,25]
            color = ['#6A0DAD','#FFC0CB','#0000FF','#90EE90','#FFFF00','#FF0000','#FFA500'] 
            hor_acc,_,_,_ = calc_pga(waveforms[0,position,0,:], waveforms[0,position,1,:], waveforms[0,position,2,:], '', 100)
            key = "%.4f,%.4f" % (float(station_coord_factor[1]), float(station_coord_factor[0]))
            file_name = stations_table_chinese[key]
            plt.figure(figsize=(20, 10))
            plt.title(f"start_index:{start_index}")
            plt.subplot(511)
            plt.plot(waveforms[0,position,0,:])
            plt.subplot(512)
            plt.plot(waveforms[0,position,1,:])
            plt.subplot(513)
            plt.plot(waveforms[0,position,2,:])
            plt.subplot(514)
            #label
            plt.plot(hor_acc)
            plt.title("Label")
            for level in range(len(pga_threshold)):
                pga_time = np.argmax(hor_acc > pga_threshold[level]*0.01*9.81)
                if (pga_time==0):
                    continue
                plt.axvline(pga_time,c=color[level],label=f"{pga_threshold[level]}*0.01*9.81")
            plt.legend()
            #pred
            plt.subplot(515)
            plt.title(f"Pred")
            plt.plot(hor_acc)
            plt.axvline(wait_list_position_dict[position]-start_index,c="g",label=f"pick")   
            plt.legend()
            if not os.path.exists(f'./img/{now_time.value}'):
                # If it doesn't exist, create it
                os.makedirs(f'./img/{now_time.value}')
            plt.savefig(f'./img/{now_time.value}/{file_name}_{first_station_position}_{position}_{station_index}.png')
            plt.clf()
            plt.close() 
            #============================================================確認輸入波型============================================================
            station_index+=1 
            
        # #(1,250,3000,3)
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
        
        # #檢查是否有新要預警的測站
        warn_Flag = False
        warning_msg=""
        log_msg = ""
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
                        print(f"Warning time: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:{target_city[city_index][0]},{target_city[city_index][-1]}\n")
                        warning_msg += f"{cnt} Warning time: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:"
                        warning_msg += f"{target_city[city_index][0]},三級:{target_city[city_index][-1][0]},四級:{target_city[city_index][-1][1]},五弱級:{target_city[city_index][-1][2]},五強級:{target_city[city_index][-1][3]},,六弱級:{target_city[city_index][-1][4]}\n"
                        log_msg += f"{cnt},{Pick_Time_dict[city_index]},{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')},{target_city[city_index][0]},{target_city[city_index][-1][0]},{target_city[city_index][-1][1]},{target_city[city_index][-1][2]},{target_city[city_index][-1][3]},{target_city[city_index][-1][4]}\n"
                        warn_Flag = True
                        cnt += 1
    
        if warn_Flag:
            # multi_station_msg_notify(warning_msg)
            logfilename_warning.value = f"./warning_log/log/{system_year}-{system_month}-{system_day}_warning.log"
            logfilename_notify.value = glob.glob("./warning_log/notify/*")
            upload_TF.value += 1
            target_city_plot.append(target_city)
            # writing picking log file
            with open(warning_logfile,"a") as pif:
                pif.write(log_msg)
                pif.close()           

     
# plotting
def WarningShower(env_config,stationInfo,key_index,warning_plot_TF,target_waveform_plot,log_name,wait_list_plot,now_time):
    
    stations_table = json.load(open(env_config["Multi_Station_Table_FILEPATH"], 'r'))
    stations_table_chinese = json.load(open(env_config["Multi_Station_Table_Chinese_FILEPATH"], 'r'))
    dataDepth = {}
    stations_table_coords = {}
    for key in stations_table.keys():
        key_sub = f"{key.split(',')[0]},{key.split(',')[1]}"
        stations_table_coords[key_sub] = stations_table[key]
        dataDepth[key_sub] = key.split(',')[2]  
    
    # create update table
    target_city_shower = {}
    for key in stations_table.keys():
        target_coord = key.split(',')
        key = "%.4f,%.4f" % (float(target_coord[0]), float(target_coord[1]))
        target_city_shower[key] = [stations_table_chinese[key],target_coord[0],target_coord[1],[0,0,0,0,0]]
            
    #============================================================參數設定============================================================
    stations_table = json.load(open(env_config["Multi_Station_Table_FILEPATH"], 'r'))
    stations_table_chinese = json.load(open(env_config["Multi_Station_Table_Chinese_FILEPATH"], 'r'))
    first_station_index = None
    lengh = (int(env_config["STORE_LENGTH"])//2)
    channel_number = 3
    time_before = 5
    #============================================================參數設定============================================================
    while True:

        # don't plot, keep pending ...
        if warning_plot_TF.value == 0.0:
            continue

        plot_filename_folder = f"./warning_log/plot/{log_name.value.split('/')[-1]}"
        msg_filename_folder = f"./warning_log/msg/{log_name.value.split('/')[-1]}"
        
        if not os.path.exists(plot_filename_folder):
            os.makedirs(plot_filename_folder)
        
        wait_list_pick_index=[]
        wait_list_position=[]
        ok_wait_list=[]
        wait_list = wait_list_plot[-1]
        scnl_list_position=[]
        station_time = []
        station_index=0
        #============================================================按照Picking順序排序============================================================
        for pick_info in wait_list:
            
            pick_info = pick_info.split(' ')
            station = pick_info[0]
            channel = pick_info[1]
            network = pick_info[2]
            location = pick_info[3]
            scnl = f"{station}_{channel}_{network}_{location}"
            station_coord_factor = get_coord_factor(np.array([scnl]), stationInfo)[0]
            depth = dataDepth[f"{station_coord_factor[1]},{station_coord_factor[0]}"]
            station_key = f"{station_coord_factor[1]},{station_coord_factor[0]},{depth}"
            
            #============================================================篩選資料============================================================
            # 同個測站不要重複
            if(f"{station}_{channel}_{network}" not in scnl_list_position):
                scnl_list_position.append(f"{station}_{channel}_{network}")
            else:
                print(f"{scnl} is duplicate")
                continue
            #============================================================篩選資料============================================================
            if(station_key in stations_table.keys()):
                position = stations_table[station_key]
                pick_index = int((float(now_time.value)-float(pick_info[10]))*100)
                wait_list_pick_index.append(pick_index)
                wait_list_position.append(position)
                ok_wait_list.append(pick_info)
                station_time.append(float(pick_info[10]))
            else:
                print(f"{station_key} not in station_table")
                pass
            station_index+=1
          
          
        #============================================================按照Picking順序排序============================================================  
        if(len(wait_list_pick_index) > 0): 
            wait_list_sort_index = np.argsort(np.array(wait_list_pick_index))
            now_index = (int(env_config["STORE_LENGTH"])//2) - int(time.time() - now_time.value)
            first_station_index = now_index - wait_list_pick_index[wait_list_sort_index[0]]
            first_station_time = station_time[wait_list_sort_index[0]]
        
        # print(f"now_time:{datetime.fromtimestamp(now_time.value)}")
        # print(f"first_station_time:{datetime.fromtimestamp(first_station_time)}")
        # print(f"now_index:{now_index}")
        # print(f"first_station_index:{first_station_index}")
        
        df = pd.read_csv(log_name.value)
        waveforms = np.zeros((1, len(stations_table),channel_number,lengh))
        station_index = 0
        for pick_info in ok_wait_list:

            station = pick_info[0]
            channel = pick_info[1]
            network = pick_info[2]
            location = pick_info[3]
            scnl = f"{station}_{channel}_{network}_{location}"
            station_coord_factor = get_coord_factor(np.array([scnl]), stationInfo)[0]
            factor = torch.Tensor(station_coord_factor[-1])
            #============================================================篩選資料============================================================
            
            # cannot search station Info
            if (station_coord_factor[1]==-1):
                print(f"{scnl} not find in stationInfo")
                wait_list.pop(station_index)
                if(len(wait_list)>0):
                    continue
                else:
                  break
              
            #確認channel命名
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
                continue
            
            #============================================================篩選資料============================================================
            #============================================================拿資料放入正確的位置============================================================
            # get waveform of z,n,e starting from ptime
            depth = dataDepth[f"{station_coord_factor[1]},{station_coord_factor[0]}"]
            station_key = f"{station_coord_factor[1]},{station_coord_factor[0]},{depth}"
            position = stations_table[station_key]
            start_index = first_station_index-time_before*100
            start_index_time = first_station_time -  time_before
            # print(f"start_index:{start_index}")
            # print(f"start_index_time:{datetime.fromtimestamp(start_index_time)}")
            pick_index = int((float(pick_info[10])-float(start_index_time))*100)
            
              
            hlz = target_waveform_plot[-1][key_index[scnl_z]][start_index:lengh]*0.01
            hln = target_waveform_plot[-1][key_index[scnl_n]][start_index:lengh]*0.01
            hle = target_waveform_plot[-1][key_index[scnl_e]][start_index:lengh]*0.01
        
            inp = torch.cat((hlz.unsqueeze(0), hln.unsqueeze(0), hle.unsqueeze(0)), dim=0)  # missing factor
            inp = inp*factor[:,None]
            inp = inp - torch.mean(inp, dim=-1, keepdims=True)
            waveforms[0,position,:,0:inp.shape[1]] = inp
            #============================================================拿資料放入正確的位置============================================================
            
            #============================================================確認輸入波型============================================================
            pga_threshold = [0.081,0.25,0.81,2.5,8.1,14,25]
            color = ['#6A0DAD','#FFC0CB','#0000FF','#90EE90','#FFFF00','#FF0000','#FFA500'] 
            hor_acc,_,_,_ = calc_pga(waveforms[0,position,0,:], waveforms[0,position,1,:], waveforms[0,position,2,:], '', 100)
            key = "%.4f,%.4f" % (float(station_coord_factor[1]), float(station_coord_factor[0]))
            target_city_row = target_city_shower[key][0].split(',')
            file_name = stations_table_chinese[key]
            plt.figure(figsize=(20, 10))
            plt.subplot(511)
            plt.plot(waveforms[0,position,0,:])
            plt.subplot(512)
            plt.plot(waveforms[0,position,1,:])
            plt.subplot(513)
            plt.plot(waveforms[0,position,2,:])
            plt.subplot(514)
            #label
            plt.plot(hor_acc)
            log_title = ["Label_0.8gal","Label_2.5gal","Label_8gal","Label_25gal","Label_81gal","Label_140gal","Label_250gal"]
            for level in range(len(pga_threshold)):
                pga_time = np.argmax(hor_acc > pga_threshold[level]*0.01*9.81)
                if (pga_time == 0):
                    continue
                label_time = datetime.fromtimestamp(start_index_time) + timedelta(seconds=int(pga_time/100)) - timedelta(seconds=time_before)
                row = df[df['Station_Id']==str(station)]
                # Description,Picking_Time,Warning_Time,Station_Id,County,Township,Station_Chinese_Name,
                # 8gal,25gal,81gal,140gal,250gal,Label_0.8gal,Label_2.5gal,Label_8gal,Label_25gal,Label_81gal,Label_140gal,Label_250gal
                if row.empty:
                    insert_row = pd.DataFrame({"Description":[f"{station_index}"], "Picking_Time":[f"{datetime.fromtimestamp(float(pick_info[10]))}"],
                                                "Station_Id":[f"{target_city_row[0]}"],"County":[f"{target_city_row[1]}"],
                                                "Township":[f"{target_city_row[2]}"],"Station_Chinese_Name":[f"{target_city_row[3]}"],
                                                "8gal":["0"],"25gal":["0"],"81gal":["0"],"140gal":["0"],"250gal":["0"]})
                    df = pd.concat([df,insert_row], ignore_index=True)
                    
                df.loc[df[df['Station_Id'] == str(station)].index, [log_title[level]]] = label_time
                plt.axvline(pga_time,c=color[level],label=f"{pga_threshold[level]}*0.01*9.81")
            plt.legend()
            #pred
            plt.subplot(515)
            plt.plot(hor_acc)
            row = df[df['Station_Id']==str(station)]
            if(len(list(row["8gal"]))!=0):
                pred = [list(row["8gal"])[0],list(row["25gal"])[0],list(row["81gal"])[0],list(row["140gal"])[0],list(row["250gal"])[0]]
                pred_level = [index for (index, item) in enumerate(pred) if item == 1]
                if (len(pred_level)==0 or (row.empty)):
                    print(f"{str(station)} is empty")
                else:
                    pred_level = [index for (index, item) in enumerate(pred) if item == 1][-1]
                    plt.title(f"Pred:{pred_level}")
                    warning_index = int((float(datetime.timestamp(datetime.strptime(f'{list(row["Warning_Time"])[0]}',"%Y-%m-%d %H:%M:%S.%f")))-float(start_index_time))*100)
                    for level in range(pred_level+1):
                        plt.axvline(warning_index,c=color[level+2],label=f"{pga_threshold[level+2]}*0.01*9.81")
                    plt.axvline(pick_index,c="g",label=f"pick")   
                    plt.legend()
            plt.savefig(f'./{plot_filename_folder}/{file_name}.png')
            plt.clf()
            plt.close() 
            station_index += 1
            #============================================================確認輸入波型============================================================
        df.to_csv(f'{log_name.value}')
        warning_plot_TF.value -= 1
        msg = output_msg(f'{log_name.value}')
        with open(f'{msg_filename_folder}', 'w') as f:
            f.write(msg)
  

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
        waveform_buffer_start_time = Value('d', nowtime.value-(int(env_config["STORE_LENGTH"])//2))
        needed_wave = []

        # a counter for accumulating key's count
        key_cnt = Value('d', int(0))
        # a dict for checking scnl's index of waveform
        key_index = manager.dict()

        # parameter for uploader
        logfilename_warning = manager.Value(c_char_p, 'hello')
        logfilename_notify = manager.Value(c_char_p, 'hello')
        log_name = manager.Value(c_char_p, 'hello')
        now_time = manager.Value(c_char_p, 'hello')
        upload_TF = Value('d', int(0))
        
        # create update table
        stations_table = json.load(open(env_config["Multi_Station_Table_Model_FILEPATH"], 'r'))
        stations_table_chinese = json.load(open(env_config["Multi_Station_Table_Chinese_FILEPATH"], 'r'))
        target_city={}
        target_city_plot = manager.list()
        target_waveform_plot = manager.list()
        wait_list  = manager.list()
        count = 0
        print("Create Table")
        for key in stations_table.keys():
            target_coord = key.split(',')
            key = "%.4f,%.4f" % (float(target_coord[0]), float(target_coord[1]))
            target_city[count] = [stations_table_chinese[key],target_coord[0],target_coord[1],[0,0,0,0,0]]
            count += 1 
        print("Finish Create Table")
        
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
        
        
        waining = Process(target=MultiStationWarning, args=(waveform_buffer, key_index,env_config,stationInfo, device,target_city,target_city_plot,
                                                            logfilename_warning,logfilename_notify,upload_TF,
                                                            warning_plot_TF,target_waveform_plot,log_name,wait_list,now_time))
        waining.start()


        # wave_shower = Process(target=WarningShower, args=(env_config,stationInfo,key_index,warning_plot_TF,
        #                                                   target_waveform_plot,log_name,wait_list,now_time))
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
        
        wave_shower.terminate()
        wave_shower.join()