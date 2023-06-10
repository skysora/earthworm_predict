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
# from multi_station_warning.multi_station_warning_util import *
from multi_station_warning.Class import *
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

def WaveSaver(env_config, waveform_buffer, key_index, nowtime, waveform_buffer_start_time, key_cnt, stationInfo,waveform_buffer_now_time,station_index):
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
    gap = 100
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
    gap=1
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
        now_time_inner = datetime.utcfromtimestamp(time.time()) +  timedelta(seconds=gap)
        now_index = int((now_time_inner - datetime.utcfromtimestamp(waveform_buffer_start_time.value/100)).seconds)*100
        toPredict_wave = cur_waveform_buffer[torch.tensor(toPredict_idx, dtype=torch.long)][:, :,now_index-3000:now_index]
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
                                                            log_name,warning_plot_TF,start_index_final_time,ok_wait_list,Pick_Time_dict,first_station_index,
                                                            first_station_time,waveforms,waveforms_final):
    
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

    
    # 記錄目前 year, month, day，用於刪除過舊的 log files
    cur = datetime.utcfromtimestamp(time.time())
    system_year, system_month, system_day = cur.year, cur.month, cur.day

    # a waiting list for model to process if multiple stations comes into PICK_RING simultaneously
    #============================================================參數設定============================================================
    wait_list = []
    cnt = 0
    First_Station_Flag = Value('d', int(0))
    dataDepth = {}
    stations_table_coords={}
    stations_table = json.load(open(env_config["Multi_Station_Table_Model_FILEPATH"], 'r'))
    stations_table_chinese = json.load(open(env_config["Multi_Station_Table_Chinese_FILEPATH"], 'r'))
    stations_table_model = json.load(open(env_config["Multi_Station_Table_Model_FILEPATH"], 'r'))
    lengh = int(env_config["PREDICT_LENGTH"])
    time_before = int(env_config["TIME_BEFORE"])
    channel_number = int(env_config["CHANNEL_NUMBER"])
    seconds_stop = int(env_config["SECOND_STOP"])
    wait_time = int(env_config["WAIT_TIME"])
    gap = int(env_config["GAP"]) 
    
    wait_list_position_dict={}
    wait_list_position_dict_time={}
    count = 0
    time_step_pred = []
    metadata = np.zeros((1, len(stations_table_model) ,int(channel_number)))
    AutoLabel_Flag = True
    dont_warning_station = [34,53,65,166,231]
    
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
        scnl_list_position=[]
        for pick_info in wait_list:
            pick_info = pick_info.split(' ')
            station_device = Device(station=pick_info[0], channel=pick_info[1],network=pick_info[2],location=pick_info[3],pick_time=datetime.utcfromtimestamp(float(pick_info[10])))
            station_coord_factor = get_coord_factor(np.array([station_device.scnl]), stationInfo)[0][0]
            station_device.set_coords(lat=station_coord_factor[0],lon=station_coord_factor[1],factor=station_coord_factor[2])
            
            #確認資料是否可以使用
            if(not station_device.is_data_available(scnl_list_position=scnl_list_position,depth_table=dataDepth,key_index_table=key_index,source=env_config["SOURCE"])):
                print(f"{station_device.scnl} is not available")
                wait_list.pop(station_index)
                if(len(wait_list)>0):
                    continue
                else:
                  break
            
            #因為不考慮高程，再前面篩選的地方有確認座標有在station table內，座標一樣的話高程用stable table內的高程取代
            depth = dataDepth[f"{station_coord_factor[1]},{station_coord_factor[0]}"]
            station_key = f"{station_coord_factor[1]},{station_coord_factor[0]},{depth}"
            position = stations_table[station_key]
            station_device.set_table_info(depth=depth,station_key=station_key,position=position)
            
            #計算index
            #每個測站的現在時間不一樣
            now_time_inner = datetime.utcfromtimestamp(time.time()) +  timedelta(seconds=gap)
            now_index = int((now_time_inner - datetime.utcfromtimestamp(waveform_buffer_start_time.value/100)).seconds)*100
            pick_index = now_index - ((now_time_inner-station_device.pick_time).seconds)*100

            
            #pick位置選擇
            if(position in wait_list_position_dict.keys()):
                if(first_station_index.value != -999 and first_station_index.value>pick_index):
                    if(abs(pick_index-first_station_index.value) <=100):
                        wait_list_position_dict[position] = pick_index
                        wait_list_position_dict_time[position] = station_device
            else:
                wait_list_position_dict[position] = pick_index
                wait_list_position_dict_time[position] = station_device
            
            station_index+=1
        
        
        ok_wait_list = list(wait_list_position_dict_time.values())
        if(len(ok_wait_list) > 0): 
            wait_list_sort_first_index = np.argsort(np.array(wait_list_position_dict.values()))[0]
            first_station_time.value = f"{ok_wait_list[wait_list_sort_first_index].pick_time}"
            first_station_index.value = int(now_index - int((now_time_inner-datetime.strptime(first_station_time.value,"%Y-%m-%d %H:%M:%S.%f")).seconds)*100)
            first_station_position = list(wait_list_position_dict.keys())[wait_list_sort_first_index]
            inner_first_station_index = first_station_index.value
            inner_first_station_time = datetime.strptime(first_station_time.value,"%Y-%m-%d %H:%M:%S.%f")
        
        # 紀錄地震開始第一個測站Picking時間
        if(len(ok_wait_list)>0 and not First_Station_Flag.value):
            First_Station_Flag.value = 1
            # get the filenames
            create_file_cur = datetime.utcfromtimestamp(time.time())
            warning_logfile = f"./warning_log/log/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}_warning.log"
            first_station_detect_time = datetime.utcfromtimestamp(time.time())
            print(f"first_station_detect_time:{first_station_detect_time}")
            with open(warning_logfile,"a") as pif:
                pif.write(f"Description,Picking_Time,Warning_Time,Station_Id,County,Township,Station_Chinese_Name,8gal,25gal,81gal,140gal,250gal,pick_time_interval,position_time_interval,pred_time_interval,update_time_interval,Label_0.8gal,Label_2.5gal,Label_8gal,Label_25gal,Label_81gal,Label_140gal,Label_250gal")
                pif.write('\n')
            if not os.path.exists(f'./warning_log/data/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}'):
                os.makedirs(f'./warning_log/data/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}')
            
            # append_data = Process(target=AppendData, args=(env_config,waveform_buffer,key_index, ok_wait_list,Pick_Time_dict,first_station_index,
                                                        
            #                                             first_station_time,waveforms_final,waveforms,First_Station_Flag,stationInfo,waveform_buffer_start_time))
            # append_data.start()
        #============================================================計算pick順序============================================================        
        if(First_Station_Flag.value and len(ok_wait_list)>0):
            diff_seconds = (datetime.utcfromtimestamp(time.time()) - first_station_detect_time).seconds
            #60秒結束判斷
            print(diff_seconds)
            if((datetime.utcfromtimestamp(time.time()) - first_station_detect_time).seconds >= (seconds_stop+wait_time)):
                print(f"reset")
                #============================================================重新驗證============================================================
                np.save(f'./warning_log/data/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}/waveform_final.npy', waveforms_final)
                np.save(f'./warning_log/data/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}/metadata_final.npy', metadata)
                np.save(f'./warning_log/data/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}/position_final.npy', np.array(list(Pick_Time_dict.keys())))
                np.save(f'./warning_log/data/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}/pick_final.npy', np.array(list(Pick_Time_dict.values())))
                #============================================================重新驗證============================================================
                log_name.value = f"{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}"
                First_Station_Flag.value = 0
                warning_plot_TF.value += 1
                count = 0
                # flush PICK_RING
                while MyModule.get_bytes(1, int(env_config["PICK_MSG_TYPE"])) != (0, 0):
                    wave_count += 1
                    continue
                wait_list=[]
                ok_wait_list = []
                wait_list_position_dict={}
                wait_list_position_dict_time={}
                Pick_Time_dict={}
                # append_data.terminate()
                waveforms = torch.zeros((1, len(stations_table_model),channel_number,int(lengh)))
                print("Create Table")
                for key in stations_table.keys():
                    target_coord = key.split(',')
                    key = "%.4f,%.4f" % (float(target_coord[0]), float(target_coord[1]))
                    target_city[count] = [stations_table_chinese[key],target_coord[0],target_coord[1],[0,0,0,0,0]]
                    count += 1 
                print("Finish Create Table")
        
        #將資料版入正確位置、預測
        if(First_Station_Flag.value and len(ok_wait_list)>0):
            pick_time_interval = (datetime.utcfromtimestamp(time.time()) - now).seconds
            now = datetime.utcfromtimestamp(time.time())
            station_index = 0
            for station_device in ok_wait_list:
                #============================================================拿資料放入正確的位置============================================================
                # get waveform of z,n,e starting from ptime
                Pick_Time_dict[position] = float(station_device.pick_time.timestamp())
                now_time_inner = datetime.utcfromtimestamp(time.time()) +  timedelta(seconds=gap)
                now_index = int((now_time_inner - datetime.utcfromtimestamp(waveform_buffer_start_time.value/100)).seconds)*100
                
                start_index = max(int(inner_first_station_index)-time_before*100,now_index-int(lengh))
                start_index_time = inner_first_station_time  - timedelta(seconds=time_before)
                start_index_final = max(int(inner_first_station_index)-time_before*100,100)
                pick_index = int((datetime.utcfromtimestamp(float(pick_info[10]))-(start_index_time)).seconds)*100
                
                hlz = waveform_buffer[key_index[station_device.scnl_z]][start_index:now_index]*0.01
                hln = waveform_buffer[key_index[station_device.scnl_n]][start_index:now_index]*0.01
                hle = waveform_buffer[key_index[station_device.scnl_e]][start_index:now_index]*0.01
                
                # draw_origin_wave(waveform_buffer[key_index[scnl_z]],waveform_buffer[key_index[scnl_n]],waveform_buffer[key_index[scnl_e]]
                #                  ,start_index,now_index,diff_seconds,"img_2",f"{position}")
                waveforms_final_inedex = now_index-start_index_final
                waveforms_final[0,position,0,:waveforms_final_inedex] =  waveform_buffer[key_index[station_device.scnl_z]][start_index_final:now_index]*0.01*station_device.factor[0]
                waveforms_final[0,position,1,:waveforms_final_inedex] =  waveform_buffer[key_index[station_device.scnl_n]][start_index_final:now_index]*0.01*station_device.factor[1]
                waveforms_final[0,position,2,:waveforms_final_inedex] =  waveform_buffer[key_index[station_device.scnl_e]][start_index_final:now_index]*0.01*station_device.factor[2]
                waveforms_final[0,position,:,:waveforms_final_inedex] = waveforms_final[0,position,:,:waveforms_final_inedex] - torch.mean(waveforms_final[0,position,:,:waveforms_final_inedex], dim=-1, keepdims=True)
                
                inp = torch.cat((hlz.unsqueeze(0), hln.unsqueeze(0), hle.unsqueeze(0)), dim=0)  # missing factor
                inp = inp*station_device.factor[:,None]
                inp = inp - torch.mean(inp, dim=-1, keepdims=True)
                #解決凸坡
                inp = slove_convex_wave(inp)
                waveforms[0,position,:,0:inp.shape[1]] = inp
                waveforms[0,position,:,inp.shape[1]-1:] = torch.mean(inp, dim=-1, keepdims=True)
                #============================================================拿資料放入正確的位置============================================================
                station_index+=1
            
            #============================================================重新驗證============================================================
            np.save(f'./warning_log/data/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}/{diff_seconds}_waveform.npy', waveforms)
            np.save(f'./warning_log/data/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}/{diff_seconds}_metadata.npy', metadata)
            np.save(f'./warning_log/data/{create_file_cur.year}-{create_file_cur.month}-{create_file_cur.day}-{create_file_cur.hour}:{create_file_cur.minute}:{create_file_cur.second}/{diff_seconds}_position.npy', np.array(list(Pick_Time_dict.keys())))
            #============================================================重新驗證============================================================
             
            position_time_interval = (datetime.utcfromtimestamp(time.time()) - now).seconds
            now = datetime.utcfromtimestamp(time.time())
            #============================================================預測============================================================ 
            
            input_waveforms = np.transpose(waveforms.cpu(),(0,1,3,2))    
            input_metadata = location_transformation(metadata)
            input_waveforms = input_waveforms.to(device)
            input_metadata = torch.Tensor(input_metadata).to(device)
            # if(diff_seconds <= seconds_stop):
            if True:
                with torch.no_grad():
                    pga_pred = model(input_waveforms,input_metadata).cpu()
                print(f"prediction finish {count} times {datetime.utcfromtimestamp(time.time())}")
                count += 1
            
            pga_times_pre = np.zeros((pga_thresholds.shape[0],pga_pred.shape[1]), dtype=int)
            
            for j,log_level in enumerate(np.log10(pga_thresholds * 9.81)):
                prob = torch.sum(
                    pga_pred[:, :, :, 0] * (1 - norm.cdf((log_level - pga_pred[:, :, :, 1]) / pga_pred[:, :, :, 2])),
                    axis=-1)
                exceedance = np.squeeze(prob > alpha[j])  # Shape: stations, 1
                pga_times_pre[j] = exceedance
                
            # #(250,5)
            pga_times_pre = np.transpose(pga_times_pre,(1,0))  
            #============================================================預測============================================================ 
            #============================================================補足預測不足的地方============================================================ 
            #如果pga真正達到目標值則預警
            station_index=0
            if(AutoLabel_Flag):
                for position in list(Pick_Time_dict.keys()):
                    pga_threshold = np.array([0.81,2.5,8.1,14,25])
                    
                    hor_acc,_,_,_ = calc_pga(input_waveforms[0,position,:,0].cpu(), input_waveforms[0,position,:,1].cpu(), input_waveforms[0,position,:,2].cpu(), '', 100)
                    for level in range(len(pga_threshold)):
                        pga_time = np.argmax(hor_acc > pga_threshold[level]*0.01*9.81)
                        if (pga_time==0):
                            continue
                        pga_times_pre[position][level] = 1
            #============================================================補足預測不足的地方============================================================ 
            
            time_step_pred.append(pga_times_pre) 
            pred_time_interval = (datetime.utcfromtimestamp(time.time()) - now).seconds
            now = datetime.utcfromtimestamp(time.time())
            # #============================================================檢查是否有新要預警的測站============================================================ 
            warn_Flag = False
            to_do_warning = np.where(pga_times_pre==True)
            densitiy = ["三級","四級","五弱級","五強級","六弱級"]
            warning_msg=""
            log_msg = ""
            pred_count = 0
            for position in to_do_warning[0]:
                if(position in dont_warning_station):
                    continue
                one_station_warn_Flag=False
                warning_msg_one_station=""
                new_target_city_list = [1]*(to_do_warning[1][pred_count]+1) + [0]*(len(densitiy)-(to_do_warning[1][pred_count]+1))
                if(target_city[position][-1] != new_target_city_list):
                    warning_msg_one_station += f"{cnt} Warning time: {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')}:{target_city[position][0]},{target_city[position][0]},"
                    for densitiy_index in range(to_do_warning[1][pred_count],-1,-1):
                        if (target_city[position][-1][densitiy_index] != new_target_city_list[densitiy_index]):
                            target_city[position][-1][densitiy_index] = new_target_city_list[densitiy_index]
                            warning_msg_one_station += f"{densitiy[densitiy_index]}:{datetime.utcfromtimestamp(time.time())}"
                            warn_Flag = True
                            one_station_warn_Flag=True
                            warning_msg_one_station+=","
                    warning_msg_one_station+="\n"
                    if(not one_station_warn_Flag):
                        warning_msg_one_station=""
                    else:
                        log_msg += f"{position},{datetime.utcfromtimestamp(float(Pick_Time_dict[position]))},{datetime.utcfromtimestamp(time.time())},{target_city[position][0]},{target_city[position][-1][0]},{target_city[position][-1][1]},{target_city[position][-1][2]},{target_city[position][-1][3]},{target_city[position][-1][4]},{pick_time_interval},{position_time_interval},{pred_time_interval},{(datetime.utcfromtimestamp(time.time()) - now).seconds}\n"
                pred_count = pred_count + 1
                warning_msg+=warning_msg_one_station

            #要大於一個測站在發布警報
            if warn_Flag:
                print(warning_msg)
                # multi_station_msg_notify(warning_msg)
                logfilename_warning.value = f"./warning_log/log/{system_year}-{system_month}-{system_day}_warning.log"
                logfilename_notify.value = glob.glob("./warning_log/notify/*")
                target_city_plot.append(target_city)
                # writing picking log file
                with open(warning_logfile,"a") as pif:
                    pif.write(log_msg)
                    pif.close()           

 
            #============================================================檢查是否有新要預警的測站============================================================
            time.sleep(0.5)
            
# plotting
def WarningShower(env_config,log_name,warning_plot_TF):
    
    stations_table_chinese = json.load(open(env_config["Multi_Station_Table_Chinese_FILEPATH_by_number"], 'r'))
    length_list=[2500,6000,8000,10000]
    while True:
        
        
        # don't plot, keep pending ...
        if warning_plot_TF.value == 0.0:
            continue

        plot_filename_folder = f"./warning_log/plot/{log_name.value}"
        msg_filename_folder = f"./warning_log/msg/{log_name.value}"
        
        if not os.path.exists(plot_filename_folder):
            os.makedirs(plot_filename_folder)
        
        # #============================================================畫畫============================================================ 
        df = pd.read_csv(f"./warning_log/log/{log_name.value}_warning.log")
        waveform = np.load(f'./warning_log/data/{log_name.value}/waveform_final.npy')
        positions = np.load(f'./warning_log/data/{log_name.value}/position_final.npy')
        pick = np.load(f'./warning_log/data/{log_name.value}/pick_final.npy')
        
    
        # length = waveform.shape[3]
        
        pick_index_list=[]
        start_time_inner = datetime.utcfromtimestamp(np.min(np.array(pick))) -  timedelta(seconds=5) - timedelta(hours=8)
        
        for position_index in range(len(positions)):
            pick_index = (datetime.utcfromtimestamp(float(pick[position_index])) - start_time_inner - timedelta(hours=8)).seconds*100
            pick_index_list.append(pick_index)
    
        
        for length in  length_list:
            station_index = 0
            for position_index in np.argsort(np.array(pick)):
                position = positions[position_index]
                file_name = stations_table_chinese[str(position)]
                color = ['#6A0DAD','#FFC0CB','#0000FF','#90EE90','#FFFF00','#FF0000','#FFA500'] 
                # sub_waveform = waveform[0,position,:,:length] - np.mean(waveform[0,position,:,:length], axis=-1, keepdims=True)
                sub_waveform = waveform[0,position,:,:length] 
                hor_acc,_,_,_ = calc_pga(sub_waveform[0],sub_waveform[1], sub_waveform[2], '', 100)
                plt.figure(figsize=(20, 10))
                plt.subplots_adjust(hspace=0.5)
                plt.subplot(511)
                plt.plot(sub_waveform[0])
                plt.subplot(512)
                plt.plot(sub_waveform[1])
                plt.subplot(513)
                plt.plot(sub_waveform[2])
                plt.subplot(514)
                # label
                plt.plot(hor_acc)
                plt.title("Label")
                answer_pga_time = {}
                pga_threshold = [0.081,0.25,0.81,2.5,8.1,14,25]
                for level in range(len(pga_threshold)):
                    pga_time = np.argmax(hor_acc > pga_threshold[level]*0.01*9.81)
                    if (pga_time==0):
                        continue
                    answer_pga_time[level]=pga_time
                    plt.axvline(pga_time,c=color[level],label=f"{pga_threshold[level]}*0.01*9.81")
                plt.legend()
                rows=df[df['Description']==position]
                #pred
                plt.subplot(515)
                plt.plot(hor_acc)
                levels=[]
                level_index=[]
                level_index_time=[]
                leading_time=[]
                for row in rows.iterrows():
                    levels =  [row[1]["8gal"],row[1]["25gal"],row[1]["81gal"],row[1]["140gal"],row[1]["250gal"]]
                    level_index_time.append(datetime.strptime(row[1]["Warning_Time"],"%Y-%m-%d %H:%M:%S.%f"))
                    print(f"start_time_inner:{start_time_inner- timedelta(hours=8)}")
                    print(datetime.strptime(row[1]["Warning_Time"],"%Y-%m-%d %H:%M:%S.%f"))
                    
                    index = (datetime.strptime(row[1]["Warning_Time"],"%Y-%m-%d %H:%M:%S.%f") - (start_time_inner - timedelta(hours=8)))*100
                    level_index.append(index.seconds)
                pga_threshold = [0.81,2.5,8.1,14,25]
                color = ['#0000FF','#90EE90','#FFFF00','#FF0000','#FFA500'] 
                leading_time_str=""
                for level in range(len(levels)):
                    if (levels[level]==0):
                        break
                    try:
                        leading_time_str += f"{int(answer_pga_time[level+2]) - int(level_index[level])},"
                    except:
                        pass
                    plt.axvline(level_index[level],c=color[level],label=f"{pga_threshold[level]}*0.01*9.81") 
                
                
                plt.title(f"label:{answer_pga_time},Pred:{level_index},diff:{leading_time_str}")
                pick_index = (datetime.utcfromtimestamp(float(pick[position_index])) - start_time_inner - timedelta(hours=8)).seconds*100
                
                plt.axvline(pick_index,c="g",label=f"pick_index") 
                plt.axvline()
                plt.legend()
                if not os.path.exists(f'./{plot_filename_folder}/{length}/'):
                    os.makedirs(f'./{plot_filename_folder}/{length}/')
                plt.savefig(f'./{plot_filename_folder}/{length}/{station_index}_{position}_{file_name}.png')
                plt.clf()
                plt.close() 
                station_index+=1
        
        warning_plot_TF.value -=1
        print(f"============================================================finish draw============================================================")

# Upload to google drive
def Uploader(logfilename_pick, logfilename_notify, logfilename_original_pick, logfilename_cwb_pick, trc_dir, upload_TF):
    print('Starting Uploader...')

    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")
    drive = GoogleDrive(gauth)

    while True:
        if upload_TF.value == 0.0:
            continue

        try:
            # upload picking log file
            if not os.path.exists(logfilename_pick.value):
                Path(logfilename_pick.value).touch()
            
            file1 = drive.CreateFile({"title":logfilename_pick.value,"parents": [{"kind": "drive#fileLink", "id": "1Y2o_Pp6np8xnxl0QysU4-zVEm4mWOI6_"}]})
            file1.SetContentFile(logfilename_pick.value)
            file1.Upload() #檔案上傳
            print("picking log file -> uploading succeeded!")

            # upload original picking log file
            if not os.path.exists(logfilename_original_pick.value):
                Path(logfilename_original_pick.value).touch()

            file1 = drive.CreateFile({"title":logfilename_original_pick.value,"parents": [{"kind": "drive#fileLink", "id": "1QmeQbsyjajpKHQXcuNxjNm426J--GZ15"}]})
            file1.SetContentFile(logfilename_original_pick.value)
            file1.Upload() #檔案上傳
            print("original picking log file -> uploading succeeded!")

            # upload notify log file
            if not os.path.exists(logfilename_notify.value):
                Path(logfilename_notify.value).touch()

            file1 = drive.CreateFile({"title":logfilename_notify.value,"parents": [{"kind": "drive#fileLink", "id": "1aqLRskDjn7Vi7WB-uzakLiBooKSe67BD"}]})
            file1.SetContentFile(logfilename_notify.value)
            file1.Upload() #檔案上傳
            print("notify log file -> uploading succeeded!")

            # upload notify log file
            if not os.path.exists(logfilename_cwb_pick.value):
                Path(logfilename_cwb_pick.value).touch()

            file1 = drive.CreateFile({"title":logfilename_cwb_pick.value,"parents": [{"kind": "drive#fileLink", "id": "1w35MfnWE3em1I0Whrd-cFn64LRKxPMBc"}]})
            file1.SetContentFile(logfilename_cwb_pick.value)
            file1.Upload() #檔案上傳
            print("CWB picker log file -> uploading succeeded!")

            upload_TF.value *= 0.0
        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Uploader): {e}\n")
                pif.write(f"Trace back (Uploader): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()
            upload_TF.value *= 0.0

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
        station_index = manager.dict()

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
                                                     waveform_buffer_now_time,station_index))
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
        stations_table_model = json.load(open(env_config["Multi_Station_Table_Model_FILEPATH"], 'r'))
        target_city={}
        target_city_plot = manager.list()
        target_waveform_plot = manager.list()
        warning_plot_TF = Value('d', int(0))
        start_index_final_time = manager.Value(c_char_p, 'hello')
        count = 0
        print("Create Table")
        for key in stations_table.keys():
            target_coord = key.split(',')
            key = "%.4f,%.4f" % (float(target_coord[0]), float(target_coord[1]))
            target_city[count] = [stations_table_chinese[key],target_coord[0],target_coord[1],[0,0,0,0,0]]
            count += 1 
        print("Finish Create Table")
        ok_wait_list = manager.list()
        Pick_Time_dict = manager.dict()
        first_station_index = Value('d', int(-999))
        first_station_time = manager.Value(c_char_p, 'hello')
        lengh = env_config["PREDICT_LENGTH"]
        channel_number = 3
        waveforms = torch.zeros((1, len(stations_table_model),channel_number,int(lengh))).share_memory_()
        waveforms_final = torch.empty((1, len(stations_table_model),channel_number, int(env_config["STORE_LENGTH"]))).share_memory_()
        # ======================================================MultiStationWarning參數======================================================
        
        waining = Process(target=MultiStationWarning, args=(waveform_buffer, key_index,env_config,waveform_buffer_start_time,stationInfo, device,target_city,
                                                            target_city_plot,logfilename_warning,logfilename_notify,
                                                            log_name,warning_plot_TF,start_index_final_time,ok_wait_list,Pick_Time_dict,first_station_index,
                                                            first_station_time,waveforms,waveforms_final))
        waining.start()

        wave_shower = Process(target=WarningShower, args=(env_config,log_name,warning_plot_TF))
        wave_shower.start()

        
        # uploader = Process(target=Uploader, args=(logfilename_warning, logfilename_notify, upload_TF))
        # uploader.start()
        
        
        wave_saver.join()
        waining.join()
        wave_shower.join()
        # uploader.join()

    except KeyboardInterrupt:
        wave_saver.terminate()
        wave_saver.join()

        waining.terminate()
        waining.join()
        
        wave_shower.terminate()
        wave_shower.join()