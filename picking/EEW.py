import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import multiprocessing
from multiprocessing import Process, Manager, Array, Value, Queue
from itertools import compress
import PyEW
import ctypes as c
import random
import pandas as pd 
import os
import glob
import bisect
import shutil

from tqdm import tqdm
import sys
sys.path.append('../')

from ctypes import c_char_p
from dotenv import dotenv_values
from datetime import datetime, timedelta
from collections import deque
from picking_preprocess import *
from picking_utils import *
from picking_model import *

import seisbench.models as sbm

# time consuming !
import matplotlib.pyplot as plt 
  
# select the stations to collect the waveform
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
def WaveSaver(env_config, waveform_buffer, key_index, nowtime, waveform_buffer_start_time, key_cnt, stationInfo, restart_cond):
    print('Starting WaveSaver...')

    MyModule = PyEW.EWModule(int(env_config["WAVE_RING_ID"]), int(env_config["PYEW_MODULE_ID"]), int(env_config["PYEW_INST_ID"]), 30.0, False)
    MyModule.add_ring(int(env_config["WAVE_RING_ID"])) # WAVE_RING
    MyModule.add_ring(int(env_config["PICK_RING_ID"])) # PICK_RING

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

    # 將所有測站切分成好幾個 chunks, 再取其中一個 chunk 作為預測的測站
    partial_station_list, _ = station_selection(sel_chunk=int(env_config["CHUNK"]), station_list=stationInfo, opt=env_config['SOURCE'], build_table=False, n_stations=int(env_config["N_PREDICTION_STATION"]))
    
    # channel mapping for tankplayer
    channel_dict = {}
    for i in range(1, 10):
        if i >= 1 and i <= 3:
            channel_dict[f"Ch{i}"] = 'FBA'
        elif i >= 4 and i <= 6:
            channel_dict[f"Ch{i}"] = 'SP'
        else:
            channel_dict[f"Ch{i}"] = 'BB'

    # 記錄系統 year, month, day, hour, minute, second，如果超過 30 秒以上沒有任何波形進來，則清空所有 buffer，等到有波形後，系統重新 pending 120s，等於是另類重啟
    system = datetime.fromtimestamp(time.time())

    while True:
        # get raw waveform from WAVE_RING
        wave = MyModule.get_wave(0) 

        if wave == {}:
            continue

        # 記錄目前 year, month, day, hour, minute, second，如果超過 30 秒以上沒有任何波形進來，則清空所有 buffer，等到有波形後，系統重新 pending 120s，等於是另類重啟
        cur = datetime.fromtimestamp(time.time())
        if (cur-system).seconds > 30:
            # restart the system
            restart_cond.value += 1
        else:
            system = cur

        if env_config['SOURCE'] == 'tankplayer':
            wave_scnl = f"{wave['station']}_{channel_dict[wave['channel']]}_{wave['network']}_{wave['location']}"

        # keep getting wave until the wave isn't empty
        if (wave['station'] not in partial_station_list and (env_config['SOURCE'] == 'Palert' or env_config['SOURCE'] == 'CWB' or env_config['SOURCE'] == 'TSMIP')) or \
                        ((env_config['SOURCE'] == 'tankplayer') and (wave_scnl not in partial_station_list)):
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
def Picker(waveform_buffer, key_index, nowtime, waveform_buffer_start_time, env_config, key_cnt, stationInfo, device,
            waveform_save_picktime, tokens, waveform_save, waveform_save_res, waveform_save_prediction, waveform_save_TF, save_info, waveform_save_waveform_starttime, 
            logfilename_pick, logfilename_original_pick, logfilename_notify, logfilename_cwb_pick, upload_TF, restart_cond, keep_wave_cnt, remove_daily,
            waveform_plot_TF, plot_info, waveform_plot_wf, waveform_plot_out, waveform_plot_picktime, plot_time):
    
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
    _, neighbor_table = station_selection(sel_chunk=int(env_config["CHUNK"]), station_list=stationInfo, opt=env_config['SOURCE'], build_table=True, n_stations=int(env_config["N_PREDICTION_STATION"]), threshold_km=float(env_config['THRESHOLD_KM']),
                                            nearest_station=int(env_config['NEAREST_STATION']), option=env_config['TABLE_OPTION'])

    # sleep 120 seconds, 讓波型先充滿 noise，而不是 0
    print('pending...')
    for _ in tqdm(range(10)):
        time.sleep(1)

    # handle line notify tokens
    line_tokens, waveform_tokens = tokens
    line_token_number, wave_token_number = 0, 0

    # use for filter the picked stations that is picked before
    pick_record = {}

    while True:
        try:
            cur = datetime.fromtimestamp(time.time())
            # 每小時發一個 notify，證明系統還活著
            if f"{system_year}-{system_month}-{system_day}-{system_hour}" != f"{cur.year}-{cur.month}-{cur.day}-{cur.hour}":
                wave_token_number = random.sample(range(len(waveform_tokens)), k=1)[0]
                wave_token_number = alive_notify(waveform_tokens, wave_token_number)
                system_hour = cur.hour

            # 已經是系統時間的隔天，檢查有沒有過舊的 log file，有的話將其刪除
            if f"{system_year}-{system_month}-{system_day}" != f"{cur.year}-{cur.month}-{cur.day}":
                toDelete_picking = cur - timedelta(days=int(env_config['DELETE_PICKINGLOG_DAY']))
                toDelete_notify = cur - timedelta(days=int(env_config['DELETE_NOTIFYLOG_DAY']))

                toDelete_picking_filename = f"./log/picking/{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_picking_chunk{env_config['CHUNK']}.log"
                toDelete_original_picking_filename = f"./log/original_picking/{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_original_picking_chunk{env_config['CHUNK']}.log"
                toDelete_notify_filename = f"./log/notify/{toDelete_notify.year}-{toDelete_notify.month}-{toDelete_notify.day}_notify_chunk{env_config['CHUNK']}.log"
                toDelete_cwbpicker_filename = f"./log/CWBPicker/{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}_picking.log"
                toDelete_exception_filename = f"./log/exception/{toDelete_picking.year}-{toDelete_picking.month}-{toDelete_picking.day}.log"

                if os.path.exists(toDelete_picking_filename):
                    os.remove(toDelete_picking_filename)
                if os.path.exists(toDelete_original_picking_filename):
                    os.remove(toDelete_original_picking_filename)
                if os.path.exists(toDelete_notify_filename):
                    os.remove(toDelete_notify_filename)
                if os.path.exists(toDelete_cwbpicker_filename):
                    os.remove(toDelete_cwbpicker_filename)
                if os.path.exists(toDelete_exception_filename):
                    os.remove(toDelete_exception_filename)

                # upload files
                logfilename_pick.value = f"./log/picking/{system_year}-{system_month}-{system_day}_picking_chunk{env_config['CHUNK']}.log"
                logfilename_original_pick.value = f"./log/original_picking/{system_year}-{system_month}-{system_day}_original_picking_chunk{env_config['CHUNK']}.log"
                logfilename_notify.value = f"./log/notify/{system_year}-{system_month}-{system_day}_notify_chunk{env_config['CHUNK']}.log"
                logfilename_cwb_pick.value = f"./log/CWBPicker/{system_year}-{system_month}-{system_day}_picking.log"

                # 把保留波形的參數歸零
                keep_wave_cnt.value *= 0
                upload_TF.value += 1
                remove_daily.value += 1

                system_year, system_month, system_day = cur.year, cur.month, cur.day

            if restart_cond.value == 1:
                # 因為即時資料中斷超過 30 秒，所以重新等待 120 seconds，等 WaveSaver 搜集波形
                print('pending...')
                for _ in tqdm(range(120)):
                    time.sleep(1)

                # log the pending 
                cur = datetime.fromtimestamp(time.time())
                picking_logfile = f"./log/picking/{cur.year}-{cur.month}-{cur.day}_picking_chunk{env_config['CHUNK']}.log"
                with open(picking_logfile,"a") as pif:
                    pif.write('='*25)
                    pif.write('\n')
                    pif.write('Real-time data interrupted 30 seconds or longer, picker pending for WaveSaver.\n')
                    pif.write('='*25)
                    pif.write('\n')
                    pif.close()

                restart_cond *= 0

            isPlot = False
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
            toPredict_wave = cur_waveform_buffer[torch.tensor(toPredict_idx, dtype=torch.long)][:, :, :3000]
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

            # 寫 original res 的 log 檔
            if np.any(original_res):
                # calculate Pa, Pv, Pd
                Pa, Pv, Pd = picking_append_info(unnormed_wave, original_res, pred_trigger)

                # calculate p_weight
                P_weight = picking_p_weight_info(out, original_res)
            
                # send pick_msg to PICK_RING
                original_pick_msg = gen_pickmsg(station_factor_coords, original_res, pred_trigger, toPredict_scnl, datetime.utcfromtimestamp(waveform_buffer_start_time.value/100), (Pa, Pv, Pd), P_weight)

                # get the filenames
                cur = datetime.fromtimestamp(time.time())
                original_picking_logfile = f"./log/original_picking/{cur.year}-{cur.month}-{cur.day}_original_picking_chunk{env_config['CHUNK']}.log"

                # writing original picking log file
                with open(original_picking_logfile,"a") as pif:
                    cur_time = datetime.utcfromtimestamp(time.time())
                    pif.write('='*25)
                    pif.write(f"Report time: {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    pif.write('='*25)
                    pif.write('\n')
                    for msg in original_pick_msg:
                        #print(msg)
                        tmp = msg.split(' ')
                        pif.write(" ".join(tmp[:6]))

                        pick_time = datetime.utcfromtimestamp(float(tmp[-5]))
                        # print('pick_time: ', pick_time)
                        pif.write(f",\tp arrival time-> {pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")

                        # write pick_msg to PICK_RING
                        # MyModule.put_bytes(1, int(env_config["PICK_MSG_TYPE"]), bytes(msg, 'utf-8'))

                    pif.close()

                # filter the picked station that picked within picktime_gap seconds before
                original_res, pick_record = check_duplicate_pick(original_res, toPredict_scnl, pick_record, pred_trigger, datetime.utcfromtimestamp(waveform_buffer_start_time.value/100), int(env_config["PICK_GAP"]))
               
                # 檢查 picking time 是否在 2500-th sample 之後
                original_res, pred_trigger, res = EEW_pick(original_res, pred_trigger)
                
                # 區域型 picking
                # res = post_picking(station_factor_coords, res, float(env_config["THRESHOLD_KM"]))                         # 用方圓幾公里來 pick
                # res = neighbor_picking(neighbor_table, station_list, res, int(env_config['THRESHOLD_NEIGHBOR']))   # 用鄰近測站來 pick

                # calculate Pa, Pv, Pd
                Pa, Pv, Pd = picking_append_info(unnormed_wave, res, pred_trigger)

                # calculate p_weight
                P_weight = picking_p_weight_info(out, res)
            
                # send pick_msg to PICK_RING
                pick_msg = gen_pickmsg(station_factor_coords, res, pred_trigger, toPredict_scnl, datetime.utcfromtimestamp(waveform_buffer_start_time.value/100), (Pa, Pv, Pd), P_weight)

                # get the filenames
                cur = datetime.fromtimestamp(time.time())
                picking_logfile = f"./log/picking/{cur.year}-{cur.month}-{cur.day}_picking_chunk{env_config['CHUNK']}.log"
                original_picking_logfile = f"./log/original_picking/{cur.year}-{cur.month}-{cur.day}_original_picking_chunk{env_config['CHUNK']}.log"

                # writing picking log file
                picked_coord = []
                with open(picking_logfile,"a") as pif:
                    cur_time = datetime.utcfromtimestamp(time.time())
                    pif.write('='*25)
                    pif.write(f"Report time: {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    pif.write('='*25)
                    pif.write('\n')
                    for msg in pick_msg:
                        #print(msg)
                        tmp = msg.split(' ')
                        pif.write(" ".join(tmp[:6]))

                        pick_time = datetime.utcfromtimestamp(float(tmp[-5]))
                        # print('pick_time: ', pick_time)
                        pif.write(f",\tp arrival time-> {pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")

                        # write pick_msg to PICK_RING
                        MyModule.put_bytes(1, int(env_config["PICK_MSG_TYPE"]), bytes(msg, 'utf-8'))

                        # filtered by P_weight
                        # p_weight = int(tmp[-4])
                        # if p_weight <= int(env_config['REPORT_P_WEIGHT']):
                        if True:
                            picked_coord.append((float(tmp[4]), float(tmp[5])))

                    pif.close()

                # plotting the station on the map and send info to Line notify
                print(f"{len(picked_coord)} stations are picked! <- {cur_time}")
                if len(picked_coord) >= int(env_config["REPORT_NUM_OF_TRIGGER"]):
                    cur_time = datetime.utcfromtimestamp(time.time())
                    trigger_plot_filename = f"{cur_time.year}-{cur_time.month}-{cur_time.day}_{cur_time.hour}:{cur_time.minute}:{cur_time.second}"
                    line_token_number = plot_taiwan(trigger_plot_filename, picked_coord, line_tokens, line_token_number)

                    # write line notify info into log file
                    picking_logfile = f"./log/notify/{system_year}-{system_month}-{system_day}_notify_chunk{env_config['CHUNK']}.log"
                    with open(picking_logfile,"a") as pif:
                        cur_time = datetime.utcfromtimestamp(time.time())
                        pif.write('='*25)
                        pif.write(f"Notify time: {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                        pif.write('='*25)
                        pif.write('\n')
                        for msg in pick_msg:
                            # print(msg)
                            tmp = msg.split(' ')
                            pif.write(" ".join(tmp[:6]))

                            pick_time = datetime.utcfromtimestamp(float(tmp[-5]))
                            pif.write(f",\tp arrival time-> {pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")

                        pif.close()

                    # plot the waveform and the prediction
                    isPlot = True
                
                # Let's save the trace!
                if waveform_save_TF.value == 0:
                    # if there is no picked station, then don't plot
                    if not np.any(original_res):
                        continue
                    
                    idx = np.arange(len(original_res))
                    tri_idx = idx[np.logical_and(original_res, nonzero_flag).tolist()]
                    
                    # # clean the save_info
                    for k in save_info.keys():
                        save_info.pop(k)

                    for iiddxx, tri in enumerate(tri_idx):
                        save_info[iiddxx] = original_pick_msg[iiddxx]
                        waveform_save[iiddxx] = original_wave[tri].cpu()
                        waveform_save_prediction[iiddxx] = out[tri].detach()

                    # nonzero_flag: 原始波形中有 0 的值為 False, 不要讓 Shower & Keeper 存波形
                    waveform_save_res[:len(original_res)] = torch.tensor(np.logical_and(original_res, nonzero_flag).tolist())
                    waveform_save_waveform_starttime.value = datetime.utcfromtimestamp(waveform_buffer_start_time.value/100).strftime('%Y-%m-%d %H:%M:%S.%f')
                    waveform_save_picktime[:len(original_res)] = torch.tensor(pred_trigger)
            
                    cur_plot_time = datetime.fromtimestamp(time.time())
                    if plot_time.value != 'Hello':
                        tmp_plot_time = datetime.strptime(plot_time.value, '%Y-%m-%d %H:%M:%S.%f')
                    if keep_wave_cnt.value < int(env_config['KEEP_WAVE_DAY']) or (plot_time.value == "Hello"):
                        if plot_time.value == "Hello":
                            waveform_save_TF.value += 1
                        else:
                            if cur_plot_time > tmp_plot_time:
                                diff = cur_plot_time - tmp_plot_time
                                if diff.total_seconds() >= 60:
                                    waveform_save_TF.value += 1

                # Let's send line notify for one of the picked stations
                if isPlot:
                    idx = np.arange(len(res))
                    tri_idx = idx[res]

                    plot_idx = random.sample(range(len(tri_idx)), k=1)[0]
                    
                    plot_info.value = pick_msg[plot_idx]
                    waveform_plot_wf[0] = toPredict_wave[tri_idx[plot_idx]].cpu()
                    waveform_plot_out[0] = out[tri_idx[plot_idx]].detach()
                    waveform_plot_picktime.value = pred_trigger[tri_idx[plot_idx]]
                    waveform_plot_TF.value += 1

            else:
                # get the filenames
                cur = datetime.fromtimestamp(time.time())
                original_picking_logfile = f"./log/original_picking/{cur.year}-{cur.month}-{cur.day}_original_picking_chunk{env_config['CHUNK']}.log"

                # writing original picking log file
                with open(original_picking_logfile,"a") as pif:
                    cur_time = datetime.utcfromtimestamp(time.time())
                    pif.write('='*25)
                    pif.write(f"Report time: {cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    pif.write('='*25)
                    pif.write('\n')
                    pif.close()

                print(f"0 stations are picked! <- {cur}")   

        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/picking/{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Picker): {e}\n")
                pif.write(f"Trace back (Picker): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()

            continue

# plotting
def Shower(waveform_plot_TF, plot_info, waveform_plot_wf, waveform_plot_out, waveform_plot_picktime, waveform_tokens):
    print('Starting Shower...')

    token_number = 0
    while True:
        # don't save the trace, keep pending ...
        if waveform_plot_TF.value == 0.0:
            continue
            
        try:            
            cur = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            
            tmp = plot_info.value.split(' ')

            # save waveform into png files
            scnl = "_".join(tmp[:4])
            savename = f"{cur}_{scnl}"
            png_filename = f"./plot/{savename}.png"

            # png title
            first_title = "_".join(tmp[:6])
            second_title = "_".join(tmp[6:10])
            p_arrival = datetime.utcfromtimestamp(float(tmp[10])).strftime('%Y-%m-%d %H:%M:%S.%f')
            other_title = "_".join(tmp[11:])
            title = f"{first_title}\n{second_title}\n{p_arrival}\n{other_title}"

            plt.figure(figsize=(12, 18))
            plt.rcParams.update({'font.size': 18})
        
            plt.subplot(411)
            plt.plot(waveform_plot_wf[0, 0])
            plt.axvline(x=waveform_plot_picktime.value, color='r')
            plt.title(title)
            
            plt.subplot(412)
            plt.plot(waveform_plot_wf[0, 1])
            plt.axvline(x=waveform_plot_picktime.value, color='r')

            plt.subplot(413)
            plt.plot(waveform_plot_wf[0, 2])
            plt.axvline(x=waveform_plot_picktime.value, color='r')

            plt.subplot(414)
            plt.ylim([-0.05, 1.05])
            plt.axvline(x=waveform_plot_picktime.value, color='r')
            plt.plot(waveform_plot_out[0])
            
            plt.savefig(png_filename)
            plt.clf()
            plt.close('all')

            token_number = random.sample(range(len(waveform_tokens)), k=1)[0]
            token_number = plot_notify(png_filename, waveform_tokens, token_number)
                    
            os.remove(png_filename)
            waveform_plot_TF.value *= 0

        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Shower): {e}\n")
                pif.write(f"Trace back (Shower): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()
            waveform_plot_TF.value *= 0
            continue

# keeping
def WaveKeeper(waveform_save, waveform_save_res, waveform_save_prediction, waveform_save_TF, save_info, waveform_save_waveform_starttime, waveform_save_picktime, keep_wave_cnt, plot_time, keep_P_weight):    
    print('Starting WaveKeeper...')

    while True:
        # don't save the trace, keep pending ...
        if waveform_save_TF.value == 0.0:
            continue
            
        cur_waveform_save_res = waveform_save_res.clone()
        waveform_starttime = waveform_save_waveform_starttime.value
        waveform_starttime = "_".join("_".join(waveform_starttime.split(':')).split('.'))
        cur_waveform_save_picktime = waveform_save_picktime.clone()

        # A directory a week
        tmp = waveform_starttime.split(' ')[0].split('-')
        year, month, day = tmp[0], tmp[1], tmp[2]
      
        day = int(day)
        if day <= 7:
            dirname = f"{year}_{month}_week1"
        elif day <= 14 and day > 7:
            dirname = f"{year}_{month}_week2"
        elif day <= 21 and day > 14:
            dirname = f"{year}_{month}_week3"
        elif day > 21:
            dirname = f"{year}_{month}_week4"
        
        if not os.path.exists(f"./plot/{dirname}"):
            os.makedirs(f"./plot/{dirname}")
        if not os.path.exists(f"./trace/{dirname}"):
            os.makedirs(f"./trace/{dirname}")

        try:
            cnt = 0
            save_plot_cnt = 0
           
            for idx, res in enumerate(cur_waveform_save_res):
                # chech the station is picked
                if res == 1:
                    tmp = save_info[cnt].split(' ')
                    
                    # if p_weight smaller than 2, then dont't save
                    if int(tmp[11]) <= int(keep_P_weight) or cur_waveform_save_picktime[idx] > 2000:
                        cnt += 1
                        continue
                    
                    # save waveform into pytorch tensor
                    scnl = "_".join(tmp[:4])
                    savename = f"{waveform_starttime}_{scnl}.pt"
                    wf = waveform_save[cnt]
                    pred = waveform_save_prediction[cnt]

                    output = {}
                    output['trace'] = wf
                    output['pred'] = pred

                    pt_filename = f"./trace/{dirname}/{savename}"
                    torch.save(output, pt_filename)

                    # save waveform into png files
                    savename = f"{waveform_starttime}_{scnl}"
                    png_filename = f"./plot/{dirname}/{savename}.png"
                    
                    # png title
                    first_title = "_".join(tmp[:6])
                    second_title = "_".join(tmp[6:10])
                    p_arrival = datetime.utcfromtimestamp(float(tmp[10])).strftime('%Y-%m-%d %H:%M:%S.%f')
                    other_title = "_".join(tmp[11:])
                    title = f"{first_title}\n{second_title}\n{p_arrival}\n{other_title}"

                    plt.figure(figsize=(12, 18))
                    plt.rcParams.update({'font.size': 18})
                
                    plt.subplot(411)
                    plt.plot(waveform_save[cnt, 0])
                    plt.axvline(x=cur_waveform_save_picktime[idx], color='r')
                    plt.title(title)
                    
                    plt.subplot(412)
                    plt.plot(waveform_save[cnt, 1])
                    plt.axvline(x=cur_waveform_save_picktime[idx], color='r')

                    plt.subplot(413)
                    plt.plot(waveform_save[cnt, 2])
                    plt.axvline(x=cur_waveform_save_picktime[idx], color='r')

                    plt.subplot(414)
                    plt.ylim([-0.05, 1.05])
                    plt.axvline(x=cur_waveform_save_picktime[idx], color='r')
                    plt.plot(waveform_save_prediction[cnt])
                    
                    plt.savefig(png_filename)
                    plt.clf()
                    plt.close('all')

                    save_plot_cnt += 1
                    cnt += 1
            
            # 這個事件沒有任何 p_weight >= 2，所以資料夾為空，刪除
            if save_plot_cnt == 0:
                pass
            else:
                keep_wave_cnt.value += 1
                plot_time.value = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')

        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Keeper): {e}\n")
                pif.write(f"Trace back (Keeper): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()
            
            continue
        
        waveform_save_TF.value *= 0

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

# remove all event in ./plot and ./trace
def Remover(remove_daily, expired_day):
    print('Starting Remover...')

    while True:
        # 檢查是不是已經試過了一天
        if remove_daily.value == 0:
            continue

        try:
            # 過期，刪除所有在 ./plot & ./trace 的事件資料夾
            folders = os.listdir('./trace')

            cur = datetime.fromtimestamp(time.time())
            cur_year, cur_month, cur_day = cur.year, cur.month, cur.day
            cur_day = int(cur_day)
            if cur_day <= 7:
                cur_day = 1
            elif cur_day <= 14 and cur_day > 7:
                cur_day = 7
            elif cur_day <= 21 and cur_day > 14:
                cur_day = 14
            elif cur_day > 21:
                cur_day = 21
            cur = datetime(year=int(cur_year), month=int(cur_month), day=cur_day)

            for f in folders:
                ymw = f.split('_')
                f_day = ymw[-1]
                if f_day == 'week4':
                    f_day = 21
                elif f_day == 'week3':
                    f_day = 14
                elif f_day == 'week2':
                    f_day = 7
                else:
                    f_day = 1
                folder_date = datetime(year=int(ymw[0]), month=int(ymw[1]), day=f_day)
                
                if (cur-folder_date).days >= int(expired_day):
                    shutil.rmtree(f"./trace/{f}")
                    shutil.rmtree(f"./plot/{f}") 

            remove_daily.value *= 0
        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (Remover): {e}\n")
                pif.write(f"Trace back (Remover): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()
            remove_daily.value *= 0

# collect the result of CWB traditional picker
def CWBPicker(env_config, trash):
    print('Starting CWBPicker...')

    MyModule = PyEW.EWModule(int(env_config["WAVE_RING_ID"]), int(env_config["PYEW_MODULE_ID"]), int(env_config["PYEW_INST_ID"]), 30.0, False)
    MyModule.add_ring(int(env_config["PICK_RING_ID"])) # PICK_RING

    while True:
        try:
            pick_msg = MyModule.get_bytes(0, int(env_config["PICK_MSG_TYPE"]))

            # if there's no data and waiting list is empty
            if pick_msg == (0, 0):
                continue

            pick_str = pick_msg[2][:pick_msg[1]].decode("utf-8").split(' ')

            # taking out the useful information to log file
            # station, channel, network, location, lontitude, latitude, p arrival time-> picking time (UTC)
            # ex. G002 HLZ SM 01 121.3039 22.9725,	p arrival time-> 2023-04-18 01:01:06.780000

            # get the filenames
            cur = datetime.fromtimestamp(time.time())
            cwb_picking_logfile = f"./log/CWBPicker/{cur.year}-{cur.month}-{cur.day}_picking.log"

            # writing CWBPicker log file
            with open(cwb_picking_logfile,"a") as pif:
                pif.write(f"{' '.join(pick_str[:6])}")

                pick_time = datetime.utcfromtimestamp(float(pick_str[10]))
                pif.write(f",\tp arrival time-> {pick_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                
                # write append information: Pa, Pv, Pd
                pif.write(f"\n(Pa, Pv, Pd, Tc)=({pick_str[6]}, {pick_str[7]}, {pick_str[8]}, {pick_str[9]})\n")
                pif.write(f"(quality, instrument, upd_sec)=({pick_str[11]}, {pick_str[12]}, {pick_str[13]})\n")
                pif.write('='*75)
                pif.write('\n')

                pif.close()
        
        except Exception as e:
            # log the pending 
            cur = datetime.fromtimestamp(time.time())
            picking_logfile = f"./log/exception/{cur.year}-{cur.month}-{cur.day}.log"
            with open(picking_logfile,"a") as pif:
                pif.write('='*25)
                pif.write('\n')
                pif.write(f"Time -> {cur.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                pif.write(f"Error message (CWBPicker): {e}\n")
                pif.write(f"Trace back (CWBPicker): {traceback.format_exc()}\n")
                pif.write('='*25)
                pif.write('\n')
                pif.close()
            
            continue


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    
    if not os.path.exists('./log/picking'):
        os.makedirs('./log/picking')
    if not os.path.exists('./log/notify'):
        os.makedirs('./log/notify')
    if not os.path.exists('./log/CWBPicker'):
        os.makedirs('./log/CWBPicker')
    if not os.path.exists('./log/original_picking'):
        os.makedirs('./log/original_picking')
    if not os.path.exists('./log/exception'):
        os.makedirs('./log/exception')
    if not os.path.exists('./plot'):
        os.makedirs('./plot')
    if not os.path.exists('./trace'):
        os.makedirs('./trace')

    try:
        # create multiprocessing manager to maintain the shared variables
        manager = Manager()
        env_config = manager.dict()
        for k, v in dotenv_values(".env").items():
            env_config[k] = v

        # a deque from time-3000 to time for time index
        nowtime = Value('d', int(time.time()*100))
        waveform_buffer_start_time = Value('d', nowtime.value-3000)

        # a counter for accumulating key's count
        key_cnt = Value('d', int(0))

        # a dict for checking scnl's index of waveform
        key_index = manager.dict()

        # restart the system in process
        restart_cond = Value('d', int(0))

        # get the station's info
        # get the station list of palert
        if env_config['SOURCE'] == 'Palert':
            stationInfo = get_PalertStationInfo(env_config['PALERT_FILEPATH'])
        elif env_config['SOURCE'] == 'CWB':
            stationInfo = get_CWBStationInfo(env_config['STAEEW_FILEPATH'])
        elif env_config['SOURCE'] == 'TSMIP':
            stationInfo = get_TSMIPStationInfo(env_config['TSMIP_FILEPATH'])
        else:
            stationInfo = get_StationInfo(env_config['NSTA_FILEPATH'], (datetime.utcfromtimestamp(time.time()) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S.%f'))

        # to save all raw wave form data, which is the numpy array, and the shape is (station numbur*channel, 3000)
        if int(env_config['CHUNK']) == -1:
            n_stations = len(stationInfo)
        else:
            n_stations = int(env_config["N_PREDICTION_STATION"])

        waveform_buffer = torch.empty((n_stations*3, int(env_config["STORE_LENGTH"]))).share_memory_()

        # waveform that should be saved in WaveKeeper
        waveform_save = torch.empty((n_stations, 3, 3000)).share_memory_()
        waveform_save_prediction = torch.empty((n_stations, 3000)).share_memory_()
        waveform_save_res = torch.empty((n_stations,)).share_memory_()
        waveform_save_picktime = torch.empty((n_stations,)).share_memory_()
        waveform_save_TF = Value('d', int(0))
        waveform_plot_TF = Value('d', int(0))
        keep_wave_cnt = Value('d', int(0))
        waveform_save_waveform_starttime = manager.Value(c_char_p, 'Hello')
        plot_time = manager.Value(c_char_p, 'Hello')
        save_info = manager.dict()

        # parameter for Shower
        plot_info = manager.Value(c_char_p, 'hello')
        waveform_plot_wf = torch.empty((1, 3, 3000)).share_memory_()
        waveform_plot_out = torch.empty((1, 3000)).share_memory_()
        waveform_plot_picktime = Value('d', int(0))

        # parameter for uploader
        logfilename_pick = manager.Value(c_char_p, 'hello')
        logfilename_notify = manager.Value(c_char_p, 'hello')
        logfilename_original_pick = manager.Value(c_char_p, 'hello')
        logfilename_cwb_pick = manager.Value(c_char_p, 'hello')
        upload_TF = Value('d', int(0))

        # parameter for remover
        remove_daily = Value('d', int(0))

        # device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        # get the candidate line notify tokens
        notify_tokens, waveform_tokens = load_tokens(env_config['NOTIFY_TOKENS'], env_config['WAVEFORM_TOKENS'])

        wave_saver = Process(target=WaveSaver, args=(env_config, waveform_buffer, key_index, nowtime, waveform_buffer_start_time, key_cnt, stationInfo, restart_cond))
        wave_saver.start()

        picker = Process(target=Picker, args=(waveform_buffer, key_index, nowtime, waveform_buffer_start_time, env_config, key_cnt, stationInfo, device,
                                                waveform_save_picktime, (notify_tokens, waveform_tokens), waveform_save, waveform_save_res, waveform_save_prediction, 
                                                waveform_save_TF, save_info, waveform_save_waveform_starttime, logfilename_pick, logfilename_original_pick, logfilename_notify, logfilename_cwb_pick, upload_TF, 
                                                restart_cond, keep_wave_cnt, remove_daily, waveform_plot_TF, plot_info, waveform_plot_wf, waveform_plot_out, waveform_plot_picktime, plot_time))
        picker.start()

        wave_shower = Process(target=Shower, args=(waveform_plot_TF, plot_info, waveform_plot_wf, waveform_plot_out, waveform_plot_picktime, waveform_tokens))
        wave_shower.start()

        wave_keeper = Process(target=WaveKeeper, args=(waveform_save, waveform_save_res, waveform_save_prediction, waveform_save_TF, save_info, 
                                                        waveform_save_waveform_starttime, waveform_save_picktime, keep_wave_cnt, plot_time, env_config['WAVE_KEEP_P_WEIGHT']))
        wave_keeper.start()

        uploader = Process(target=Uploader, args=(logfilename_pick, logfilename_notify, logfilename_original_pick, logfilename_cwb_pick, env_config['TRC_PATH'], upload_TF))
        uploader.start()

        remover = Process(target=Remover, args=(remove_daily, env_config['WAVE_EXPIRED']))
        remover.start()

        cwbpicker_logger = Process(target=CWBPicker, args=(env_config, 0))
        cwbpicker_logger.start()

        wave_saver.join()
        picker.join()
        wave_shower.join()
        wave_keeper.join()
        uploader.join()
        remover.join()
        cwbpicker_logger.join()

    except KeyboardInterrupt:
        wave_saver.terminate()
        wave_saver.join()

        picker.terminate()
        picker.join()

        wave_shower.terminate()
        wave_shower.join()

        wave_keeper.terminate()
        wave_keeper.join()

        uploader.terminate()
        uploader.join()

        remover.terminate()
        remover.join()

        cwbpicker_logger.terminate()
        cwbpicker_logger.join()
        