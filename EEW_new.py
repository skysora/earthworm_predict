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
import json
import csv
from cala import *
from decimal import Decimal
import sys
import seisbench.models as sbm

#multi-station
import multi_station_warning.models as models
from multi_station_warning.util import *
#Picker
sys.path.append('./picking')
from picking_preprocess import *
from picking_utils import *
from picking_model import *

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/TEAM/lib
# CUDA_VISIBLE_DEVICES=2 python3 EEW.py
# picking: pick and send pick_msg to PICK_RING
def distance(coord1, coord2):
    R = 6373.0

    lon1, lat1 = radians(coord1[0]), radians(coord1[1])
    lon2, lat2 = radians(coord2[0]), radians(coord2[1])

    dlon, dlat = lon2-lon1, lat2-lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    distance = R * c
    return distance


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


def station_selection(sel_chunk, station_list, opt, build_table=False, n_stations=None, threshold_km=None,
                    nearest_station=3, option='nearest'):
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
    
        # if sel_chunk == -1, then collect all station in TSMIP
        if sel_chunk == -1:
            output_chunks = []
            for k, v in station_list.items():
                output_chunks.append((k, [float(v[0]), float(v[1]), float(v[2])]))
            output_chunks = [output_chunks]

        table = 0
        if build_table:
            # build the table that contains every station in "threshold_km" km for each station
            table = build_neighborStation_table(output_chunks[sel_chunk], threshold_km, nearest_station, option)

        output = []
        for o in output_chunks[sel_chunk]:
            output.append(o[0])

        return output, table

    elif opt == 'TSMIP':
        lon_split = np.array([120.91])
        lat_split = np.array([21.9009 , 24.03495, 26.169  ])

        # 依照經緯度切分測站
        chunk = [[] for _ in range(6)]
        for k, sta in station_list.items():
            row, col = 0, 0

            row = bisect.bisect_left(lon_split, float(sta[0]))
            col = bisect.bisect_left(lat_split, float(sta[1]))

            chunk[2*col+row].append((k, [float(sta[0]), float(sta[1]), float(sta[2])]))

        output_chunks = []
        output_chunks.append(chunk[3])
        
        chunk[2] = sorted(chunk[2], key = lambda x : x[1][1])
        output_chunks.append(chunk[2][len(chunk[2])//2:])
        output_chunks.append(chunk[2][:len(chunk[2])//2])
        output_chunks[-1] += chunk[0]
        
        chunk[5] = sorted(chunk[5], key = lambda x : x[1][0])
        output_chunks.append(chunk[5][:50] + chunk[4])
        output_chunks.append(chunk[5][50:])

        new_output_chunks2 = []
        for sta in output_chunks[2]:
            if sta[1][1] <= 22.5:
                output_chunks[0].append(sta)
            else:
                new_output_chunks2.append(sta)
        output_chunks[2] = new_output_chunks2

        new_output_chunks1 = []
        for sta in output_chunks[1]:
            if sta[1][1] >= 23.977:
                output_chunks[3].append(sta)
            else:
                new_output_chunks1.append(sta)
        output_chunks[1] = new_output_chunks1

        # if sel_chunk == -1, then collect all station in TSMIP
        if sel_chunk == -1:
            output_chunks = []
            for k, v in station_list.items():
                output_chunks.append((k, [float(v[0]), float(v[1]), float(v[2])]))
            output_chunks = [output_chunks]
       
        table = 0
        if build_table:
            # build the table that contains every station in "threshold_km" km for each station
            table = build_neighborStation_table(output_chunks[sel_chunk], threshold_km, nearest_station, option)
                
        output = []
        for o in output_chunks[sel_chunk]:
            output.append(o[0])

        return output, table

    else:
        # sort the station_list by lontitude
        stationInfo = sorted(station_list.items(), key = lambda x : x[1][0])

        if sel_chunk == -1:
            n_stations = len(stationInfo)

        station_chunks = [stationInfo[n_stations*i:n_stations*i+n_stations] 
                            for i in range(len(stationInfo)//n_stations)]
        
        table = 0
        if build_table:
            # build the table that contains every station in "threshold_km" km for each station
            table = build_neighborStation_table(station_chunks[sel_chunk], threshold_km, nearest_station, option)

        return [i[0] for i in station_chunks[sel_chunk]], table


# to save wave form from WAVE_RING
def WaveSaver(env_config, waveform_buffer, key_index, nowtime, waveform_buffer_start_time, key_cnt, stationInfo,waveform_buffer_now_time):
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
    stations_table = json.load(open(env_config["Multi_Station_Table_Model_FILEPATH"], 'r'))
    dataDepth = {}
    stations_table_coords = {}
    for key in stations_table.keys():
        key_sub = f"{key.split(',')[0]},{key.split(',')[1]}"
        stations_table_coords[key_sub] = stations_table[key]
        dataDepth[key_sub] = key.split(',')[2]
    cnt = 0
    gap = 500
    while True:
        # get raw waveform from WAVE_RING
        wave = MyModule.get_wave(0) 
        
        # keep getting wave until the wave isn't empty
        # if wave != {}:
                # print(wave['station'] not in partial_station_list)
        # if wave == {} or (wave['station'] not in partial_station_list and (env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'TSMIP')):
        if wave == {}:    
            continue
        # print(f"wave:{wave}")
        station = wave['station']
        channel = wave['channel']
        network = wave['network']
        location = wave['location']
        startt = wave['startt']
        nsamp = wave['nsamp']
        endt = wave['endt']
        scnl = f"{station}_{channel}_{network}_{location}"
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
            
            waveform_buffer[key_index[scnl]][startIndex:startIndex+nsamp] = torch.from_numpy(wave['data'].copy().astype(np.float32))
            waveform_buffer_now_time[key_index[scnl]] = time.time()
        
            # if(network == "SMT" and channel[-1]=="1"):
            #     if not os.path.exists(f'./img_1/{scnl}'):
            #         # If it doesn't exist, create it
            #         os.makedirs(f'./img_1/{scnl}')
            #     plt.plot(waveform_buffer[key_index[scnl]][:])
            #     plt.axvline(position[0][-1])
            #     plt.savefig(f'./img_1/{scnl}/{datetime.utcfromtimestamp(time.time())}.png')
            #     plt.clf()
            
        except Exception as e:
            print(e)
            print(f"{scnl} can't assign wave data into waveform")
            print(key_index[scnl], startIndex, startIndex+nsamp)
        
        # move the time window of timeIndex and waveform every 5 seconds
        if int(time.time()*100) - nowtime.value >= gap:
            waveform_buffer_start_time.value += gap
            waveform_buffer[:, 0:int(env_config["STORE_LENGTH"])-gap] = waveform_buffer[:, gap:int(env_config["STORE_LENGTH"])]
            
            # the updated waveform is fill in with 0
            waveform_buffer[:, int(env_config["STORE_LENGTH"])-gap:int(env_config["STORE_LENGTH"])] = torch.zeros((waveform_buffer.shape[0],gap))
            nowtime.value += gap     


def Picker(waveform_buffer, key_index, nowtime, waveform_buffer_start_time, env_config, key_cnt, stationInfo, device):
    
    print('Starting Picker...')
    MyModule = PyEW.EWModule(int(env_config["WAVE_RING_ID"]), int(env_config["PYEW_MODULE_ID"]), int(env_config["PYEW_INST_ID"]), 30.0, False)
    MyModule.add_ring(int(env_config["WAVE_RING_ID"])) # WAVE_RING
    MyModule.add_ring(int(env_config["PICK_RING_ID"])) # PICK_RING

    # conformer picker
    model_path = env_config["PICKER_CHECKPOINT_PATH"]
    if env_config["CHECKPOINT_TYPE"] == 'conformer':
        model = SingleP_Conformer(conformer_class=16, d_ffn=256, n_head=4, enc_layers=4, dec_layers=4, d_model=12, encoder_type='conformer', decoder_type='crossattn').to(device)
    elif env_config["CHECKPOINT_TYPE"] == 'conformer_other':
        model = SingleP_Conformer(conformer_class=16, d_ffn=256, n_head=4, enc_layers=4, dec_layers=4, d_model=12, encoder_type='conformer', decoder_type='crossattn', label_type='other').to(device)
    elif env_config["CHECKPOINT_TYPE"] == 'eqt':
        model = sbm.EQTransformer(in_samples=3000).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    channel_tail = ['Z', 'N', 'E']
    PICK_MSG_TYPE = int(env_config["PICK_MSG_TYPE"])    

    # 記錄目前 year, month, day，用於刪除過舊的 log files
    cur = datetime.fromtimestamp(time.time())
    system_year, system_month, system_day = cur.year, cur.month, cur.day
    system_hour = cur.hour
    
    # Neighbor_table: 蒐集每個測站方圓 X km 內所有測站的代碼
    # _, neighbor_table = station_selection(sel_chunk=int(env_config["CHUNK"]), station_list=stationInfo, opt=env_config['SOURCE'], build_table=True, n_stations=int(env_config["N_PREDICTION_STATION"]), threshold_km=float(env_config['THRESHOLD_KM']),
    #                                         nearest_station=int(env_config['NEAREST_STATION']), option=env_config['TABLE_OPTION'])

    # sleep 120 seconds, 讓波型先充滿 noise，而不是 0
    # print('pending...')
    # for _ in tqdm(range(10)):
    #     time.sleep(1)

    # use for filter the picked stations that is picked before
    pick_record = {}

    while True:
        cur = datetime.fromtimestamp(time.time())
        cur_waveform_buffer, cur_key_index = waveform_buffer.clone(), key_index.copy()
        
        # skip if there is no waveform in buffer or key_index is collect faster than waveform
        if int(key_cnt.value) == 0 or key_cnt.value < len(key_index):
            continue
    
        # collect the indices of stations that contain 3-channel waveforms
        toPredict_idx, VtoA_idx, toPredict_scnl = [], [], []
        for k, v in cur_key_index.items():
                tmp = k.split('_')

                if env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'TSMIP':
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
        
        toPredict_wave = cur_waveform_buffer[torch.tensor(toPredict_idx, dtype=torch.long)][:, :, int(env_config["STORE_LENGTH"])//2-3000:int(env_config["STORE_LENGTH"])//2]
        toPredict_scnl = np.array(toPredict_scnl)
        VtoA_idx = np.array(VtoA_idx)
        
        # get the factor and coordinates of stations
        if env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'TSMIP':
            station_factor_coords, station_list = get_Palert_CWB_coord(toPredict_scnl, stationInfo)

            # count to gal
            factor = np.array([f[-1] for f in station_factor_coords]).astype(float)
            toPredict_wave = toPredict_wave/factor[:, None, None]
        else:
            station_factor_coords, station_list = get_coord_factor(toPredict_scnl, stationInfo)

            # multiply with factor to convert count to 物理量
            factor = np.array([f[-1] for f in station_factor_coords])
            toPredict_wave = toPredict_wave*factor[:, :, None]
           
        # preprocess
        # 1) convert traces to acceleration
        # 2) 1-45Hz bandpass filter
        # 3) Z-score normalization
        # 4) calculate features: Characteristic, STA, LTA

        # original wave: used for saving waveform
        original_wave = toPredict_wave.clone()
        toPredict_wave = v_to_a(toPredict_wave, VtoA_idx)
        unnormed_wave = toPredict_wave.clone()
        toPredict_wave, _ = z_score(toPredict_wave)
        _, nonzero_flag = z_score(original_wave.clone())
        toPredict_wave = filter(toPredict_wave)
        if env_config["CHECKPOINT_TYPE"] == 'conformer' or env_config["CHECKPOINT_TYPE"] == 'conformer_other':
            toPredict_wave = calc_feats(toPredict_wave)

        #  predict
        toPredict_wave = torch.FloatTensor(toPredict_wave).to(device)
        with torch.no_grad():
            # for conformer
            if env_config["CHECKPOINT_TYPE"] == 'conformer':
                out = model(toPredict_wave).squeeze().cpu()   
            elif env_config["CHECKPOINT_TYPE"] == 'conformer_other':
                out = model(toPredict_wave).squeeze().cpu()   
                out = out[:, :, 0]
            # for eqt
            elif env_config["CHECKPOINT_TYPE"] == 'eqt':
                out = model(toPredict_wave)[1].squeeze().cpu() 
        plotidx = np.random.randint(low=0, high=out.shape[0])
        
        # plt.subplot(211)
        # plt.plot(toPredict_wave[plotidx].T.cpu())
        # plt.subplot(212)
        # plt.ylim([-0.05, 1.05])
        # plt.plot(out[plotidx].detach().numpy())
        # plt.savefig(f'/home/sora/M11015203/docker-home/Earthworm/pyearthworm-predict-pga25/img_pick/{datetime.fromtimestamp(time.time())}_{toPredict_scnl[plotidx]}.png')
        # plt.clf()
        # reduce the result in order to speed up the process
        # only reduce the stations when chunk == -1 
        if int(env_config['CHUNK']) == -1:
            out = out[:int(env_config['N_PREDICTION_STATION'])]
            station_list = station_list[:int(env_config['N_PREDICTION_STATION'])]
            station_factor_coords = station_factor_coords[:int(env_config['N_PREDICTION_STATION'])]
            toPredict_wave = toPredict_wave[:int(env_config['N_PREDICTION_STATION'])]
            original_wave = original_wave[:int(env_config['N_PREDICTION_STATION'])]
            unnormed_wave = unnormed_wave[:int(env_config['N_PREDICTION_STATION'])]
            toPredict_scnl = toPredict_scnl[:int(env_config['N_PREDICTION_STATION'])]
            nonzero_flag = nonzero_flag[:int(env_config['N_PREDICTION_STATION'])]

        # select the p-arrival time 
        original_res, pred_trigger = evaluation(out, float(env_config["THRESHOLD_PROB"]), int(env_config["THRESHOLD_TRIGGER"]), env_config["THRESHOLD_TYPE"])
        # print(original_res)

        # # 寫 original res 的 log 檔
        if np.any(original_res):
        #     # calculate Pa, Pv, Pd
        #     Pa, Pv, Pd = picking_append_info(unnormed_wave, original_res, pred_trigger)

        #     # calculate p_weight
        #     P_weight = picking_p_weight_info(out, original_res)
        
        #     # send pick_msg to PICK_RING
        #     original_pick_msg = gen_pickmsg(station_factor_coords, original_res, pred_trigger, toPredict_scnl, datetime.utcfromtimestamp(waveform_buffer_start_time.value/100), (Pa, Pv, Pd), P_weight)

        #     # get the filenames
        #     cur = datetime.fromtimestamp(time.time())
        #     original_picking_logfile = f"./log/original_picking/{cur.year}-{cur.month}-{cur.day}_original_picking_chunk{env_config['CHUNK']}.log"

        #     # writing original picking log file
        #     with open(original_picking_logfile,"a") as pif:
        #         cur_time = datetime.utcfromtimestamp(time.time())
        #         pif.write('='*25)
        #         pif.write(f"Report time: {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        #         pif.write('='*25)
        #         pif.write('\n')
        #         for msg in original_pick_msg:
        #             #print(msg)
        #             tmp = msg.split(' ')
        #             pif.write(" ".join(tmp[:6]))

        #             pick_time = datetime.utcfromtimestamp(float(tmp[-5]))
        #             # print('pick_time: ', pick_time)
        #             pif.write(f",\tp arrival time-> {pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")

        #             # write pick_msg to PICK_RING
        #             # MyModule.put_bytes(1, int(env_config["PICK_MSG_TYPE"]), bytes(msg, 'utf-8'))

        #         pif.close()

        #     # filter the picked station that picked within picktime_gap seconds before
            original_res, pick_record = check_duplicate_pick(original_res, toPredict_scnl, pick_record, pred_trigger, datetime.utcfromtimestamp(waveform_buffer_start_time.value/100), int(env_config["PICK_GAP"]))
            
        #     # 檢查 picking time 是否在 2500-th sample 之後
            original_res, pred_trigger, res = EEW_pick(original_res, pred_trigger)
            
        #     # 區域型 picking
        #     # res = post_picking(station_factor_coords, res, float(env_config["THRESHOLD_KM"]))                         # 用方圓幾公里來 pick
        #     # res = neighbor_picking(neighbor_table, station_list, res, int(env_config['THRESHOLD_NEIGHBOR']))   # 用鄰近測站來 pick

        #     # calculate Pa, Pv, Pd
            Pa, Pv, Pd = picking_append_info(unnormed_wave, res, pred_trigger)

        #     # calculate p_weight
            P_weight = picking_p_weight_info(out, res)
        
        #     # send pick_msg to PICK_RING
            pick_msg = gen_pickmsg(station_factor_coords, res, pred_trigger, toPredict_scnl, datetime.utcfromtimestamp((waveform_buffer_start_time.value/100)+70), (Pa, Pv, Pd), P_weight)

        #     # get the filenames
        #     cur = datetime.fromtimestamp(time.time())
        #     picking_logfile = f"./log/picking/{cur.year}-{cur.month}-{cur.day}_picking_chunk{env_config['CHUNK']}.log"
        #     original_picking_logfile = f"./log/original_picking/{cur.year}-{cur.month}-{cur.day}_original_picking_chunk{env_config['CHUNK']}.log"

        #     # writing picking log file
        #     picked_coord = []
            for msg in pick_msg:
        #         #print(msg)
                tmp = msg.split(' ')

                # pick_time = datetime.utcfromtimestamp(float(tmp[-5]))
        #         # print('pick_time: ', pick_time)

        #         # write pick_msg to PICK_RING
                MyModule.put_bytes(1, int(env_config["PICK_MSG_TYPE"]), bytes(msg, 'utf-8'))

        #         # filtered by P_weight
        #         # p_weight = int(tmp[-4])
        #         # if p_weight <= int(env_config['REPORT_P_WEIGHT']):
        #         if True:
        #             picked_coord.append((float(tmp[4]), float(tmp[5])))

        # else:
        #     print(f"0 stations are picked! <- {cur}")   

        # except Exception as e:
        #     continue

def MultiStationWarning(waveform_buffer, key_index,env_config,waveform_buffer_start_time,stationInfo, device,target_city,
                        target_city_plot,logfilename_warning,logfilename_notify,
                        wait_list_plot,target_waveform_plot,log_name,wait_list,nowtime,warning_plot_TF,waveform_buffer_now_time):
    
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
    cur = datetime.utcfromtimestamp(time.time())
    system_year, system_month, system_day = cur.year, cur.month, cur.day

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
    first_station_index = None
    lengh = 3000
    channel_number = 3
    seconds_stop = 30
    time_before = 5
    gap = 5 #seconds
    wait_list_position_dict={}
    wait_list_position_dict_time={}
    count = 0
    wait_time = 30
    time_step_pred = []
    metadata = np.zeros((1, len(stations_table_model) ,channel_number))
    waveforms = np.zeros((1, len(stations_table_model),channel_number,lengh))
    waveforms_final = torch.empty((1, len(stations_table_model),channel_number, int(env_config["STORE_LENGTH"])//2))
    AutoLabel_Flag = False
    for key in stations_table.keys():
        position = stations_table[key]
        metadata[0,position] = np.array([key.split(',')[0],key.split(',')[1],key.split(',')[2]])
        key_sub = f"{key.split(',')[0]},{key.split(',')[1]}"
        stations_table_coords[key_sub] = stations_table[key]
        dataDepth[key_sub] = key.split(',')[2]    
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
            pick_str_list = pick_msg[2][:pick_msg[1]].decode("utf-8").split(" ")
            # the quality of wave should not greater than 2
            if int(pick_str_list[11]) > 0:
                continue

            # the using time period for picking module should be 3 seconds
            # if int(pick_str_list[13])  != 3:
            #     continue
            wait_list.append(pick_str)
    
        #============================================================計算pick順序============================================================
        station_index = 0
        now = datetime.utcfromtimestamp(time.time())
        # now_time_inner = datetime.utcfromtimestamp(float(nowtime.value/100)) +  timedelta(seconds=gap)
        # now_time_inner = datetime.utcfromtimestamp(time.time()) +  timedelta(seconds=gap)
        # now_index = int((now_time_inner - datetime.utcfromtimestamp(waveform_buffer_start_time.value/100)).seconds)*100
        
        scnl_list_position=[]
        for pick_info in wait_list:

            pick_info = pick_info.split(' ')
            station = pick_info[0]
            channel = pick_info[1]
            network = pick_info[2]
            location = pick_info[3]
            quality = pick_info[-3]
            scnl = f"{station}_{channel}_{network}_{location}"
            station_coord_factor = get_coord_factor(np.array([scnl]), stationInfo)[0][0]
            #============================================================篩選資料============================================================
            # cannot search station Info
            if (station_coord_factor[0]==-1):
                print(f"{scnl} not find in stationInfo")
                # wait_list.pop(station_index)
                if(len(wait_list)>0):
                    continue
                else:
                  break
            # 同個測站不要重複
            if(f"{station}_{channel}_{network}" not in scnl_list_position):
                scnl_list_position.append(f"{station}_{channel}_{network}")
            else:
                print(f"{scnl} is duplicate")
                # wait_list.pop(station_index)
                if(len(wait_list)>0):
                    continue
                else:
                  break
            # 資料不在table內
            if(f"{station_coord_factor[1]},{station_coord_factor[0]}" not in dataDepth):
                print(f"{scnl} not find in stationInfo")
                # wait_list.pop(station_index)
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
                # print(f"{scnl} one of 3 channel missing")
                continue
            #============================================================篩選資料============================================================
            
            depth = dataDepth[f"{station_coord_factor[1]},{station_coord_factor[0]}"]
            station_key = f"{station_coord_factor[1]},{station_coord_factor[0]},{depth}"
            position = stations_table[station_key]
            #每個測站的現在時間不一樣
            now_time_inner = datetime.utcfromtimestamp(time.time()) +  timedelta(seconds=gap)
            now_index = int((now_time_inner - datetime.utcfromtimestamp(waveform_buffer_start_time.value/100)).seconds)*100
            pick_index = now_index - ((now_time_inner-datetime.utcfromtimestamp(float(pick_info[10]))).seconds)*100
    
            #pick位置選擇
            if(station_key in stations_table.keys()):
                if(position in wait_list_position_dict.keys()):
                    if(first_station_index is not None and first_station_index>pick_index):
                        if(abs(pick_index-first_station_index) <=100):
                            wait_list_position_dict[position] = pick_index
                            wait_list_position_dict_time[position] = pick_info
                else:
                    wait_list_position_dict[position] = pick_index
                    wait_list_position_dict_time[position] = pick_info
            else:
                print(f"{station_key} not in station_table")
                pass

            station_index+=1
        
        
        ok_wait_list = list(wait_list_position_dict_time.values())
        if(len(ok_wait_list) > 0): 
            wait_list_sort_first_index = np.argsort(np.array(wait_list_position_dict.values()))[0]
            first_station_time = datetime.utcfromtimestamp(float(ok_wait_list[wait_list_sort_first_index][10]))
            first_station_index = now_index - int((now_time_inner-first_station_time).seconds)*100
            first_station_position = list(wait_list_position_dict.keys())[wait_list_sort_first_index]
        
        
        # #紀錄地震開始第一個測站Picking時間
        if(len(ok_wait_list)>0 and not First_Station_Flag):
            First_Station_Flag = True
            # get the filenames
            create_file_cur = datetime.utcfromtimestamp(time.time())
            warning_logfile = f"./warning_log/log/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}_warning.log"
            first_station_detect_time = datetime.utcfromtimestamp(time.time())
            print(f"first_station_detect_time:{first_station_detect_time}")
            with open(warning_logfile,"a") as pif:
                pif.write(f"Description,Picking_Time,Warning_Time,Station_Id,County,Township,Station_Chinese_Name,8gal,25gal,81gal,140gal,250gal,pick_time_interval,position_time_interval,pred_time_interval,update_time_interval,Label_0.8gal,Label_2.5gal,Label_8gal,Label_25gal,Label_81gal,Label_140gal,Label_250gal")
                pif.write('\n')
        
        #============================================================計算pick順序============================================================        
        #將資料版入正確位置、預測
        if(First_Station_Flag and len(ok_wait_list)>0):
            #60秒結束判斷
            diff_seconds = (datetime.utcfromtimestamp(time.time()) - first_station_detect_time).seconds
            print(diff_seconds)
            if((datetime.utcfromtimestamp(time.time()) - first_station_detect_time).seconds >= seconds_stop):
                print(f"reset")
                #============================================================重新驗證============================================================
                np.save(f'./warning_log/data/{system_year}-{system_month}-{system_day}_waveform_final.npy', waveforms_final)
                np.save(f'./warning_log/data/{system_year}-{system_month}-{system_day}_metadata_final.npy', metadata)
                np.save(f'./warning_log/data/{system_year}-{system_month}-{system_day}_position_final.npy', np.array(list(Pick_Time_dict.keys())))
                #============================================================重新驗證============================================================
                for _ in tqdm(range(wait_time)):
                    time.sleep(1)
                
                First_Station_Flag = False
                log_name.value = warning_logfile
                target_waveform_plot.append(waveform_buffer)
                wait_list_plot.append(wait_list.copy())
                warning_plot_TF.value += 1
                wait_list=[]
                ok_wait_list = []
                wait_list_position_dict={}
                wait_list_position_dict_time={}
                count = 0
                for _ in tqdm(range(wait_time)):
                    time.sleep(1)
                    
                print("Create Table")
                for key in stations_table.keys():
                    target_coord = key.split(',')
                    key = "%.4f,%.4f" % (float(target_coord[0]), float(target_coord[1]))
                    target_city[count] = [stations_table_chinese[key],target_coord[0],target_coord[1],[0,0,0,0,0]]
                    count += 1 
                print("Finish Create Table")
        
            pick_time_interval = (datetime.utcfromtimestamp(time.time()) - now).seconds
            now = datetime.utcfromtimestamp(time.time())
            #============================================================資料到正確位置============================================================
                
            station_index = 0
            scnl_list_position = []
            for pick_info in ok_wait_list:
                
                station = pick_info[0]
                channel = pick_info[1]
                network = pick_info[2]
                location = pick_info[3]
                scnl = f"{station}_{channel}_{network}_{location}"
                station_coord_factor = get_coord_factor(np.array([scnl]), stationInfo)[0][0]
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
                Pick_Time_dict[position] = datetime.utcfromtimestamp(float(pick_info[10]))
                
                now_time_inner = datetime.utcfromtimestamp(time.time()) +  timedelta(seconds=gap)
                now_index = int((now_time_inner - datetime.utcfromtimestamp(waveform_buffer_start_time.value/100)).seconds)*100
                
                start_index = max(first_station_index-time_before*100,now_index-lengh)
                start_index_time = first_station_time  - timedelta(seconds=time_before)
                pick_index = int((datetime.utcfromtimestamp(float(pick_info[10]))-(start_index_time)).seconds)*100
                
                hlz = waveform_buffer[key_index[scnl_z]][start_index:now_index]*0.01
                hln = waveform_buffer[key_index[scnl_n]][start_index:now_index]*0.01
                hle = waveform_buffer[key_index[scnl_e]][start_index:now_index]*0.01

                waveforms_final[0,position,0,:now_index-(first_station_index-time_before*100)] =  waveform_buffer[key_index[scnl_z]][(first_station_index-time_before*100):now_index]*0.01*factor[0,None]
                waveforms_final[0,position,1,:now_index-(first_station_index-time_before*100)] =  waveform_buffer[key_index[scnl_n]][(first_station_index-time_before*100):now_index]*0.01*factor[1,None]
                waveforms_final[0,position,2,:now_index-(first_station_index-time_before*100)] =  waveform_buffer[key_index[scnl_e]][(first_station_index-time_before*100):now_index]*0.01*factor[2,None]
                waveforms_final[0,position,:,:now_index-(first_station_index-time_before*100)] = waveforms_final[0,position,:,:now_index-(first_station_index-time_before*100)] - torch.mean(waveforms_final[0,position,:,:now_index-(first_station_index-time_before*100)], dim=-1, keepdims=True)
                # waveforms_final[0,position,:,:] = slove_convex_wave(torch.Tensor(waveforms_final[0,position,:,:])).cpu()
                
                inp = torch.cat((hlz.unsqueeze(0), hln.unsqueeze(0), hle.unsqueeze(0)), dim=0)  # missing factor
                inp = inp*factor[:,None]
                inp = inp - torch.mean(inp, dim=-1, keepdims=True)
                #解決凸坡
                inp = slove_convex_wave(inp)
                waveforms[0,position,:,0:inp.shape[1]] = inp
                waveforms[0,position,:,inp.shape[1]-1:] = torch.mean(inp, dim=-1, keepdims=True)
                # sub_waveform = torch.Tensor(waveforms[0,position,:,:])
                # waveforms[0,position,:,:] = slove_convex_wave(sub_waveform).cpu()
                # waveforms = waveforms - np.mean(waveforms, axis=-1, keepdims=True)
                #============================================================重新驗證============================================================
                np.save(f'./warning_log/data/{system_year}-{system_month}-{system_day}_{diff_seconds}_waveform.npy', waveforms)
                np.save(f'./warning_log/data/{system_year}-{system_month}-{system_day}_{diff_seconds}_metadata.npy', metadata)
                np.save(f'./warning_log/data/{system_year}-{system_month}-{system_day}_{diff_seconds}_position.npy', np.array(list(Pick_Time_dict.keys())))
                #============================================================重新驗證============================================================
                #============================================================拿資料放入正確的位置============================================================
                station_index+=1 
            
            #============================================================資料到正確位置============================================================  
            position_time_interval = (datetime.utcfromtimestamp(time.time()) - now).seconds
            now = datetime.utcfromtimestamp(time.time())
            #============================================================預測============================================================ 
            
            # input_waveforms = np.transpose(waveforms,(0,1,3,2))    
            # input_metadata = location_transformation(metadata)
            # input_waveforms = torch.Tensor(input_waveforms).to(device)
            # input_metadata = torch.Tensor(input_metadata).to(device)
            # with torch.no_grad():
            #     pga_pred = model(input_waveforms,input_metadata).cpu()
            
            # pga_times_pre = np.zeros((pga_thresholds.shape[0],pga_pred.shape[1]), dtype=int)
                
            # for j,log_level in enumerate(np.log10(pga_thresholds * 9.81)):
            #     prob = torch.sum(
            #         pga_pred[:, :, :, 0] * (1 - norm.cdf((log_level - pga_pred[:, :, :, 1]) / pga_pred[:, :, :, 2])),
            #         axis=-1)
            #     exceedance = np.squeeze(prob > alpha[j])  # Shape: stations, 1
            #     pga_times_pre[j] = exceedance
                
            # #(250,5)
            # pga_times_pre = np.transpose(pga_times_pre,(1,0))  
            # #============================================================預測============================================================ 
            # #============================================================補足預測不足的地方============================================================ 
            # #如果pga真正達到目標值則預警
            # station_index=0
            # if(AutoLabel_Flag):
            #     for position in list(Pick_Time_dict.keys()):
            #         now_time_inner = datetime.utcfromtimestamp(time.time())
            #         # +  timedelta(seconds=gap)
                    
            #         pga_threshold = np.array([0.81,2.5,8.1,14,25])
            #         hor_acc,_,_,_ = calc_pga(input_waveforms[0,position,:,0].cpu(), input_waveforms[0,position,:,1].cpu(), input_waveforms[0,position,:,2].cpu(), '', 100)
            #         for level in range(len(pga_threshold)):
            #             pga_time = np.argmax(hor_acc > pga_threshold[level]*0.01*9.81)
            #             if (pga_time==0):
            #                 continue
            #             pga_times_pre[position][level] = 1
                        
            #         #============================================================畫畫============================================================
            #         pga_threshold = [0.081,0.25,0.81,2.5,8.1,14,25]
            #         color = ['#6A0DAD','#FFC0CB','#0000FF','#90EE90','#FFFF00','#FF0000','#FFA500'] 
            #         hor_acc,_,_,_ = calc_pga(input_waveforms[0,position,:,0].cpu(), input_waveforms[0,position,:,1].cpu(), input_waveforms[0,position,:,2].cpu(), '', 100)
            #         key = "%.4f,%.4f" % (float(station_coord_factor[1]), float(station_coord_factor[0]))
            #         file_name = stations_table_chinese[key]
            #         plt.figure(figsize=(20, 10))
            #         plt.subplots_adjust(hspace=0.5)
            #         plt.subplot(511)
            #         plt.title(f"start_index:{start_index},pick_time:{datetime.utcfromtimestamp(float(wait_list_position_dict_time[position][10]))}")
            #         plt.plot(input_waveforms[0,position,:,0].cpu())
            #         plt.subplot(512)
            #         plt.plot(input_waveforms[0,position,:,1].cpu())
            #         plt.subplot(513)
            #         plt.plot(input_waveforms[0,position,:,2].cpu())
            #         plt.subplot(514)
            #         #label
            #         plt.plot(hor_acc)
            #         plt.title("Label")
            #         for level in range(len(pga_threshold)):
            #             pga_time = np.argmax(hor_acc > pga_threshold[level]*0.01*9.81)
            #             if (pga_time==0):
            #                 continue
            #             plt.axvline(pga_time,c=color[level],label=f"{pga_threshold[level]}*0.01*9.81")
            #         plt.legend()
            #         #pred
            #         plt.subplot(515)
            #         plt.title(f"Now time:{now_time_inner}")
            #         plt.plot(hor_acc)
            #         pick_index = int((Pick_Time_dict[position]-start_index_time).seconds)*100
            #         now_index_in_3000 = int((now_time_inner  - start_index_time).seconds)*100
            #         pga_threshold = [0.81,2.5,8.1,14,25]
            #         plt.axvline(pick_index,c="g",label=f"pick")
            #         plt.axvline(now_index_in_3000,c="r",label=f"now_ position")
            #         # color = ['#0000FF','#90EE90','#FFFF00','#FF0000','#FFA500'] 
            #         # for level in range(len(pga_threshold)):
            #         #     if (pga_times_pre[position][level]!=1):
            #         #         continue
            #         #     plt.axvline(now_index_in_3000,c=color[level],label=f"{pga_threshold[level]}*0.01*9.81")
            #         # plt.axvline(pick_index,c="g",label=f"pick")   
            #         plt.legend()
            #         if not os.path.exists(f'./img_1/{diff_seconds}'):
            #             # If it doesn't exist, create it
            #             os.makedirs(f'./img_1/{diff_seconds}')
            #         plt.savefig(f'./img_1/{diff_seconds}/{station_index}_{scnl}_{first_station_position}_{position}.png')
            #         plt.clf()
            #         plt.close() 
            #         station_index+=1 
            #         #============================================================畫畫============================================================
                
            # #============================================================補足預測不足的地方============================================================ 
            
            # time_step_pred.append(pga_times_pre) 
            # pred_time_interval = (datetime.utcfromtimestamp(time.time()) - now).seconds
            # now = datetime.utcfromtimestamp(time.time())
            # #============================================================檢查是否有新要預警的測站============================================================ 
            # warn_Flag = False
            # warning_msg=""
            # log_msg = ""
            
            # for city_index in range(pga_times_pre.shape[0]):
            #     #update 表格
            #     if(set(target_city[city_index][-1]) != set(pga_times_pre[city_index])):
            #         indices = [index for (index, item) in enumerate(pga_times_pre[city_index]) if item ==1 ]
            #         for warning_thresholds in range(len(pga_thresholds)):
            #             if(warning_thresholds in indices):
            #                 target_city[city_index][-1][warning_thresholds] += 1
            #             else:
            #                 target_city[city_index][-1][warning_thresholds] += 0
                        
            #         if (len(indices)!=0):
            #             Flag = True
            #             for indice in indices:
            #                 for index in range(indice,-1,-1):
            #                     if(target_city[city_index][-1][index]==0):
            #                         #不預警
            #                         Flag=False
            #                 if (not Flag) :
            #                     target_city[city_index][-1][indice] -= 1
            #                 if(Flag) and (target_city[city_index][-1][indice]>1):
            #                     Flag = False
                                
            #             if Flag:
            #                 print(f"Warning time: {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:{target_city[city_index][0]},{target_city[city_index][-1]}\n")
            #                 warning_msg += f"{cnt} Warning time: {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:"
            #                 warning_msg += f"{target_city[city_index][0]},三級:{target_city[city_index][-1][0]},四級:{target_city[city_index][-1][1]},五弱級:{target_city[city_index][-1][2]},五強級:{target_city[city_index][-1][3]},六弱級:{target_city[city_index][-1][4]}\n"
            #                 log_msg += f"{cnt},{Pick_Time_dict[city_index]},{datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')},{target_city[city_index][0]},{target_city[city_index][-1][0]},{target_city[city_index][-1][1]},{target_city[city_index][-1][2]},{target_city[city_index][-1][3]},{target_city[city_index][-1][4]},{pick_time_interval},{position_time_interval},{pred_time_interval},{(datetime.utcfromtimestamp(time.time()) - now).seconds}\n"
            #                 warn_Flag = True
            #                 cnt += 1
        
            # if warn_Flag:
            #     # multi_station_msg_notify(warning_msg)
            #     logfilename_warning.value = f"./warning_log/log/{system_year}-{system_month}-{system_day}_warning.log"
            #     logfilename_notify.value = glob.glob("./warning_log/notify/*")
            #     target_city_plot.append(target_city)
            #     # writing picking log file
            #     with open(warning_logfile,"a") as pif:
            #         pif.write(log_msg)
            #         pif.close()           

            # print(f"prediction finish {count} times {datetime.utcfromtimestamp(time.time())}")
            # count += 1 
            #============================================================檢查是否有新要預警的測站============================================================
            
# plotting
def WarningShower(waveform_buffer_start_time,waveform_buffer,env_config,stationInfo,key_index,wait_list_plot,target_waveform_plot,log_name,wait_list,nowtime,warning_plot_TF):
    
    
    #============================================================參數設定============================================================
    first_station_index = None
    lengh = (int(env_config["STORE_LENGTH"])//2)
    channel_number = 3
    time_before = 10
    stations_table = json.load(open(env_config["Multi_Station_Table_Model_FILEPATH"], 'r'))
    stations_table_chinese = json.load(open(env_config["Multi_Station_Table_Chinese_FILEPATH"], 'r'))
    dataDepth = {}
    stations_table_coords = {}
    first_station_index = None
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
        
    wait_list_position_dict={}
    wait_list_position_dict_time={}
    #============================================================參數設定============================================================
    while True:
        
        
        # don't plot, keep pending ...
        if warning_plot_TF.value == 0.0:
            continue

        plot_filename_folder = f"./warning_log/plot/{log_name.value.split('/')[-1]}"
        msg_filename_folder = f"./warning_log/msg/{log_name.value.split('/')[-1]}"
        
        if not os.path.exists(plot_filename_folder):
            os.makedirs(plot_filename_folder)
        
        
        #============================================================按照Picking順序排序============================================================ 
        station_index = 0
        ok_wait_list = []
        scnl_list_position=[]
        wait_list = wait_list_plot[-1].copy()
        
        for pick_info in wait_list:

            pick_info = pick_info.split(' ')
            station = pick_info[0]
            channel = pick_info[1]
            network = pick_info[2]
            location = pick_info[3]
            quality = pick_info[-3]
            scnl = f"{station}_{channel}_{network}_{location}"
            station_coord_factor = get_coord_factor(np.array([scnl]), stationInfo)[0][0]
            #============================================================篩選資料============================================================
            # cannot search station Info
            if (station_coord_factor[0]==-1):
                print(f"{scnl} not find in stationInfo")
                # wait_list.pop(station_index)
                if(len(wait_list)>0):
                    continue
                else:
                  break
            # 同個測站不要重複
            if(f"{station}_{channel}_{network}" not in scnl_list_position):
                scnl_list_position.append(f"{station}_{channel}_{network}")
            else:
                print(f"{scnl} is duplicate")
                # wait_list.pop(station_index)
                if(len(wait_list)>0):
                    continue
                else:
                  break
            # 資料不在table內
            if(f"{station_coord_factor[1]},{station_coord_factor[0]}" not in dataDepth):
                print(f"{scnl} not find in stationInfo")
                # wait_list.pop(station_index)
                if(len(wait_list)>0):
                    continue
                else:
                  break
            #============================================================篩選資料============================================================
            
            depth = dataDepth[f"{station_coord_factor[1]},{station_coord_factor[0]}"]
            station_key = f"{station_coord_factor[1]},{station_coord_factor[0]},{depth}"
            position = stations_table[station_key]
            now_time_inner = datetime.utcfromtimestamp(time.time())
            now_index = int((now_time_inner - datetime.utcfromtimestamp(waveform_buffer_start_time.value/100)).seconds)*100
            pick_index = now_index - int(float(datetime.timestamp((now_time_inner))-float(pick_info[10]))*100)
            #pick位置選擇
            
            if(station_key in stations_table.keys()):
                if(position in wait_list_position_dict.keys()):
                    # if(first_station_index is not None and first_station_index>pick_index):
                    wait_list_position_dict[position] = pick_index
                    wait_list_position_dict_time[position] = pick_info
                else:
                    wait_list_position_dict[position] = pick_index
                    wait_list_position_dict_time[position] = pick_info
            else:
                print(f"{station_key} not in station_table")
                pass

            station_index+=1
        


        ok_wait_list = list(wait_list_position_dict_time.values())
        if(len(ok_wait_list) > 0): 
            wait_list_sort_first_index = np.argsort(np.array(wait_list_position_dict.values()))[0]
            first_station_time = datetime.utcfromtimestamp(float(ok_wait_list[wait_list_sort_first_index][10]))
            first_station_index = now_index - int((now_time_inner-first_station_time).seconds)*100
            first_station_position = list(wait_list_position_dict.keys())[wait_list_sort_first_index]
        

        #============================================================按照Picking順序排序============================================================  
        #============================================================畫畫============================================================ 
        df = pd.read_csv(log_name.value)
        waveforms = np.zeros((1, len(stations_table),channel_number,lengh))
        station_index = 0
        for pick_info in ok_wait_list:

            station = pick_info[0]
            channel = pick_info[1]
            network = pick_info[2]
            location = pick_info[3]
            scnl = f"{station}_{channel}_{network}_{location}"
            station_coord_factor = get_coord_factor(np.array([scnl]), stationInfo)[0][0]
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
            # get waveform of z,n,e starting from ptime
            depth = dataDepth[f"{station_coord_factor[1]},{station_coord_factor[0]}"]
            station_key = f"{station_coord_factor[1]},{station_coord_factor[0]},{depth}"
            position = stations_table[station_key]
            
            now_time_inner = datetime.utcfromtimestamp(time.time())
            now_index = int((now_time_inner - datetime.utcfromtimestamp(waveform_buffer_start_time.value/100)).seconds)*100
            
            start_index = max(first_station_index-time_before*100,now_index-lengh)
            start_index_time = first_station_time  - timedelta(seconds=time_before)
            pick_index = int((datetime.utcfromtimestamp(float(pick_info[10]))-(start_index_time)).seconds)*100
            
            
            hlz = waveform_buffer[key_index[scnl_z]][start_index:now_index]*0.01
            hln = waveform_buffer[key_index[scnl_n]][start_index:now_index]*0.01
            hle = waveform_buffer[key_index[scnl_e]][start_index:now_index]*0.01
        
            inp = torch.cat((hlz.unsqueeze(0), hln.unsqueeze(0), hle.unsqueeze(0)), dim=0)  # missing factor
            inp = inp*factor[:,None]
            inp = inp - torch.mean(inp, dim=-1, keepdims=True)
            #解決凸坡
            inp = slove_convex_wave(inp)
            waveforms[0,position,:,0:inp.shape[1]] = inp
            sub_waveform = torch.Tensor(waveforms[0,position,:,:])
            waveforms[0,position,:,:] = slove_convex_wave(sub_waveform).cpu()
            
            #============================================================確認輸入波型============================================================
            pga_threshold = [0.081,0.25,0.81,2.5,8.1,14,25]
            color = ['#6A0DAD','#FFC0CB','#0000FF','#90EE90','#FFFF00','#FF0000','#FFA500'] 
            hor_acc,_,_,_ = calc_pga(waveforms[0,position,0,:], waveforms[0,position,1,:], waveforms[0,position,2,:], '', 100)
            key = "%.4f,%.4f" % (float(station_coord_factor[1]), float(station_coord_factor[0]))
            target_city_row = target_city_shower[key][0].split(',')
            file_name = stations_table_chinese[key]
            
            plt.figure(figsize=(20, 10))
            plt.subplots_adjust(hspace=0.5)
            # plt.tight_layout()
            plt.subplot(511)
            plt.title(f"{scnl}")
            plt.plot(waveforms[0,position,0,:])
            plt.subplot(512)
            plt.plot(waveforms[0,position,1,:])
            plt.subplot(513)
            plt.plot(waveforms[0,position,2,:])
            plt.subplot(514)
            #label
            plt.plot(hor_acc)
            log_title = ["Label_0.8gal","Label_2.5gal","Label_8gal","Label_25gal","Label_81gal","Label_140gal","Label_250gal"]
            label_msg=""
            for level in range(len(pga_threshold)):
                pga_time = np.argmax(hor_acc > pga_threshold[level]*0.01*9.81)
                if (pga_time == 0):
                    continue
                label_msg = f"pga_time:{pga_time},pick_index:{pick_index},diff:{pga_time-pick_index},"
                label_time = start_index_time + timedelta(seconds=int(pga_time/100))
                row = df[df['Station_Id']==str(station)]
                # Description,Picking_Time,Warning_Time,Station_Id,County,Township,Station_Chinese_Name,
                # 8gal,25gal,81gal,140gal,250gal,Label_0.8gal,Label_2.5gal,Label_8gal,Label_25gal,Label_81gal,Label_140gal,Label_250gal
                if row.empty:
                    print(target_city_row)
                    try:
                        insert_row = pd.DataFrame({"Description":[f"{station_index}"], "Picking_Time":[f"{datetime.utcfromtimestamp(float(pick_info[10]))}"],
                                                    "Station_Id":[f"{target_city_row[0]}"],"County":[f"{target_city_row[1]}"],
                                                    "Township":f"{key}","Station_Chinese_Name":f"{key}",
                                                    "8gal":["0"],"25gal":["0"],"81gal":["0"],"140gal":["0"],"250gal":["0"]})
                    except:
                        insert_row = pd.DataFrame({"Description":[f"{station_index}"], "Picking_Time":[f"{datetime.utcfromtimestamp(float(pick_info[10]))}"],
                                                    "Station_Id":[f"{target_city_row[0]}"],"County":"",
                                                    "Township":f"{key}","Station_Chinese_Name":f"{key}",
                                                    "8gal":["0"],"25gal":["0"],"81gal":["0"],"140gal":["0"],"250gal":["0"]})
                    df = pd.concat([df,insert_row], ignore_index=True)
                    
                df.loc[df[df['Station_Id'] == str(station)].index, [log_title[level]]] = label_time
                plt.axvline(pga_time,c=color[level],label=f"{pga_threshold[level]}*0.01*9.81")
            plt.title(label_msg)
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
                    warning_index = int((float(datetime.timestamp(datetime.strptime(f'{list(row["Warning_Time"])[0]}',"%Y-%m-%d %H:%M:%S.%f")))-float(datetime.timestamp(start_index_time)))*100)
                    for level in range(pred_level+1):
                        plt.axvline(warning_index,c=color[level+2],label=f"{pga_threshold[level+2]}*0.01*9.81")          
                    plt.title(f'warning_index:{warning_index},pick_index:{pick_index},diff:{warning_index-pick_index},')
            plt.axvline(pick_index,c="g",label=f"pick")   
            plt.legend()
            plt.savefig(f'./{plot_filename_folder}/{station_index}_{file_name}_{first_station_position}_{position}.png')
            plt.clf()
            plt.close() 
            station_index += 1
            #============================================================確認輸入波型============================================================
        print(f"============================================================finish draw============================================================")
        df.to_csv(f'{log_name.value}')
        warning_plot_TF.value -= 1
        msg = output_msg(f'{log_name.value}')
        with open(f'{msg_filename_folder}', 'w') as f:
            f.write(msg)
        #============================================================畫畫============================================================ 

if __name__ == '__main__':
    
    #basic code setting
    torch.multiprocessing.set_start_method('spawn')
    
    try:
        # ======================================================共用參數======================================================
        # create multiprocessing manager to maintain the shared variables
        manager = Manager()
        env_config = manager.dict()
        for k, v in dotenv_values(".env").items():
            env_config[k] = v
        #device
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
        upload_TF = Value('d', int(0))
        # to save all raw wave form data, which is the numpy array, and the shape is (station numbur*channel, 3000)
        waveform_buffer = torch.empty((int(env_config["N_PREDICTION_STATION"])*3, int(env_config["STORE_LENGTH"]))).share_memory_()
        waveform_buffer_now_time = torch.empty((int(env_config["N_PREDICTION_STATION"])*3)).share_memory_()
        # ======================================================共用參數======================================================
        
        
            
        wave_saver = Process(target=WaveSaver, args=(env_config, waveform_buffer, key_index, nowtime, waveform_buffer_start_time, key_cnt, stationInfo,
                                                     waveform_buffer_now_time))
        wave_saver.start()
        # ======================================================Picker======================================================
        # picker = Process(target=Picker, args=(waveform_buffer, key_index, nowtime, waveform_buffer_start_time, env_config, key_cnt, stationInfo, device))
        # picker.start()
        # ======================================================Picker======================================================
        # ======================================================MultiStationWarning參數======================================================
        #folder create
        if not os.path.exists('./warning_log/log'):
            os.makedirs('./warning_log/log')
        if not os.path.exists('./warning_log/plot'):
            os.makedirs('./warning_log/plot')
        if not os.path.exists('./warning_log/msg'):
            os.makedirs('./warning_log/msg')
        if not os.path.exists('./warning_log/data'):
            os.makedirs('./warning_log/data')
        
        # create update table
        stations_table = json.load(open(env_config["Multi_Station_Table_Model_FILEPATH"], 'r'))
        stations_table_chinese = json.load(open(env_config["Multi_Station_Table_Chinese_FILEPATH"], 'r'))
        target_city={}
        target_city_plot = manager.list()
        target_waveform_plot = manager.list()
        wait_list  = manager.list()
        wait_list_plot = manager.list()
        warning_plot_TF = Value('d', int(0))
        count = 0
        print("Create Table")
        for key in stations_table.keys():
            target_coord = key.split(',')
            key = "%.4f,%.4f" % (float(target_coord[0]), float(target_coord[1]))
            target_city[count] = [stations_table_chinese[key],target_coord[0],target_coord[1],[0,0,0,0,0]]
            count += 1 
        print("Finish Create Table")
        # ======================================================MultiStationWarning參數======================================================
        
        waining = Process(target=MultiStationWarning, args=(waveform_buffer, key_index,env_config,waveform_buffer_start_time,stationInfo, device,target_city,
                                                            target_city_plot,logfilename_warning,logfilename_notify,
                                                            wait_list_plot,target_waveform_plot,log_name,wait_list,nowtime,warning_plot_TF,waveform_buffer_now_time))
        waining.start()

        wave_shower = Process(target=WarningShower, args=(waveform_buffer_start_time,waveform_buffer,env_config,stationInfo,key_index,wait_list_plot,target_waveform_plot,
                                                          log_name,wait_list,nowtime,warning_plot_TF))
        wave_shower.start()

        
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