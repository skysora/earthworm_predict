import numpy as np
import json 
import glob
import pandas as pd
import os
import math
import argparse
import bisect
from datetime import datetime, timedelta
from staticmap import StaticMap, CircleMarker, Polygon, Line
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_stationlist(sel_chunk):
    tsmip_path = '/home/weiwei/disk4/pyearthworm-predict-pga25/picking/nsta/sta_csmt_Z'
    with open(tsmip_path, 'r') as f:
        sta_eew = f.readlines()

    stationInfo = {}
    for l in sta_eew:
        tmp = l.split()
        stationInfo[tmp[0]] = [tmp[5], tmp[4], tmp[-2]]

    station_list = station_selection(sel_chunk, stationInfo)
    return station_list

def station_selection(sel_chunk, station_list):
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
    
    output = []
    for o in output_chunks[sel_chunk]:
        output.append(o[0])
        
    return output

def find_all_cwbsnCode(sta_list, mapping):
    cwbsn_code = []
    for sta in sta_list:
        try:
            tmp = TSMIP_NSTA_mapping(mapping, sta+' ')
            if type(tmp) == str:
                cwbsn_code.append(tmp)
        except Exception as e:
            continue
    
    return cwbsn_code

def search(log, epicenter, search_date, pass_sec, plot):
    compare = {}
    p_arr = {}
    origin_date = search_date
    for i in tqdm(range(pass_sec)):
        if i != 0:
            search_date = search_date + timedelta(seconds=1)
            
        to_search = search_date.strftime('%Y-%m-%d %H:%M:%S')
        if i == 0:
            plot_dir = to_search.replace(':', '_') 
            if not os.path.exists(f"./plot/{plot_dir}"):
                os.makedirs(f"./plot/{plot_dir}")
        to_search = f"=Report time: {to_search}"

        info = []
        isFind = False
        for k, v in log.items():
            for line in v:
                if isFind:
                    if '====' not in line:
                        info.append(line)
                    else:
                        if to_search not in line:
                            break

                if to_search in line:
                    isFind = True
                    continue
        
        if not isFind:
            continue

        # print('-'*50)
        # print('nowtime: ', search_date)
        compare[search_date] = []
        for tmp in info:
            p_arrival = tmp.split('->')[-1].strip()
            date_tmp = p_arrival.split(' ')
            ymd, hms = date_tmp[0], date_tmp[1]
            yy = ymd.split('-')
            hh = hms.split(':')
            sta = tmp.split(' ')[0]
            p_arrival = datetime(year=int(yy[0]), month=int(yy[1]), day=int(yy[2]), hour=int(hh[0]), minute=int(hh[1]), second=int(math.floor(float(hh[2]))))
            if p_arrival > origin_date:
                compare[search_date].append(sta)

                if sta not in p_arr:
                    p_arr[sta] = tmp.split('->')[-1].strip()
        # print('-'*50)

        if plot:
            plot_taiwan(info, epicenter, to_search, plot_dir)
        
    return compare, p_arr

def plot_taiwan(info, epicenter, date, plot_dir):
    sta, coord = [], []
    for l in info:
        tmp = l.split(' ')
        sta.append(tmp[0])
        coord.append((float(tmp[4]), float(tmp[5].split(',')[0])))
        
    # plot
    m = StaticMap(300, 400)
    
    # epicenter
    marker_outline = CircleMarker(epicenter, 'blue', 12)
    m.add_marker(marker_outline)
    
    for c in coord:
        marker = CircleMarker(c, '#eb4034', 8)

        m.add_marker(marker)
    
    image = m.render(zoom=7)
    image.save(f"./plot/{plot_dir}/{date.replace(':', '_')}.png")

def TSMIP_NSTA_mapping(mapping, station):
    sta = mapping.loc[mapping['TSMIP_short_code'] == f"{station}"].CWBSN_code.to_list()[0]
    
    return sta

def CWBSN_NSTA_mapping(mapping, station):
    sta = mapping.loc[mapping['CWBSN_code'] == f"{station}"].TSMIP_short_code.to_list()[0]
    
    return sta

def open_P(pfile_path, sta_list, opt):
    pfile = {}
    for sta in sta_list:
        pfile[sta] = (0, 0)
   
    if os.path.exists(pfile_path):
        print('Open pfile: ', pfile_path)
        with open(pfile_path) as f:
            lines = f.readlines()
    else:
        print("No pfile found!")
        return pfile        
    
    for i in lines[1:]:
        minute = int(i[21:23])
        second = math.floor(float(i[23:29]))

        if second >= 60:
            minute += 1
            second -= 60
            
        sta = i[:5].strip()
        if sta in sta_list:
            pfile[sta] = (minute, second)
        
    return pfile

def find_pfilename(opt, pfiledir):
    year, month, day, hour, minute, second = opt.year, opt.month, opt.day, opt.hour, opt.minute, opt.second

    if minute >= 10 and hour-8 >= 10:
        pfile_path = f"{month+12}{day}{hour-8}{minute}.*{str(year)[-2:]}" 
    elif minute < 10 and hour-8 >= 10:
        pfile_path = f"{month+12}{day}{hour-8}0{minute}.*{str(year)[-2:]}" 
    elif minute >= 10 and hour-8 < 10:
        pfile_path = f"{month+12}{day}0{hour-8}{minute}.*{str(year)[-2:]}" 
    elif minute < 10 and hour-8 < 10:
        pfile_path = f"{month+12}{day}0{hour-8}0{minute}.*{str(year)[-2:]}" 

    pfile_list = glob.glob(f"{pfiledir}{pfile_path}")
    if len(pfile_list) == 0:
        pfile_date = datetime(year=year, month=month, day=day, hour=hour-8, minute=minute)
        possible_pfile = [pfile_date - timedelta(minutes=1), pfile_date + timedelta(minutes=1)]
    else:
        return pfile_list
    
    for p in possible_pfile:
        month = str(p.month + 12)
        day = str(p.day) if p.day >= 10 else "0"+str(p.day)
        hour = str(p.hour) if p.hour >= 10 else "0"+str(p.hour)
        minute = str(p.minute) if p.minute >= 10 else "0"+str(p.minute)
        
        tmp_filename = f"{pfiledir}{month+day+hour+minute}.P{int(year % 100)}"

        if os.path.exists(tmp_filename):
            with open(tmp_filename) as f:
                lines = f.readlines()

            # compare event origin time
            tmp = lines[0]

            year = int(tmp[1:5])
            month = int(tmp[5:7])
            day = int(tmp[7:9])
            hour = int(tmp[9:11])
            minute = int(tmp[11:13])
            
            if year == opt.year and month == opt.month and day == opt.day and hour+8 == opt.hour and minute == opt.minute:
                tmp = tmp_filename[:-3]
                pfile_list = glob.glob(f"{tmp}*{int(year % 100)}")
                break

    return pfile_list

def find_first_pick(compare, pfile, savepath, p_arr):
    with open(savepath, 'w') as f:
        f.write('Results\n')

    time_diff = 0.0
    cnt = 0
    fp, fn = 0, 0
    tp, tn = 0, 0
    for station in tqdm(pfile.keys(), total=len(list(pfile.keys()))):
        isPick = False
        try:
            tsmip_sta = CWBSN_NSTA_mapping(mapping, station).strip()
            for k, v in compare.items():
                if tsmip_sta in v:
                    pfile_min, pfile_sec = pfile[station]
                    
                    if pfile_min == 0 and pfile_sec == 0:
                        with open(savepath, 'a') as f:
                            f.write(f"station: {tsmip_sta}/{station}\n")
                            f.write(f"AI picker picking time-> {k}\n")
                            f.write(f"AI picker p-wave arrival time-> {p_arr[tsmip_sta]}\n")
                            f.write(f"Pfile -> X\n")
                            f.write(f"AI picker is False Positive\n")
                            f.write('-'*50)
                            f.write('\n')
                            fp += 1
                            
                            isPick = True
                        break
                        
                    pfile_time = k.replace(minute=pfile_min, second=pfile_sec)
                    
                    with open(savepath, 'a') as f:
                        f.write(f"station: {tsmip_sta}/{station}\n")
                        f.write(f"AI picker picking time-> {k}\n")
                        f.write(f"AI picker p-wave arrival time-> {p_arr[tsmip_sta]}\n")
                        f.write(f"Pfile -> {pfile_time}\n")

                        if k>pfile_time:
                            f.write(f"AI picker delay {(k-pfile_time).seconds} second(s)\n")
                            time_diff += (k-pfile_time).seconds
                        else:
                            f.write(f"AI picker is earlier about {(pfile_time-k).seconds} second(s)\n")
                            time_diff += ((pfile_time-k).seconds)*-1
                        f.write('-'*50)
                        f.write('\n')
                    
                    tp += 1
                    cnt += 1
                    isPick = True
                    break
                    
            if not isPick:
                if pfile_min == 0 and pfile_sec == 0:
                    tn += 1
                else:
                    with open(savepath, 'a') as f:
                        f.write(f"station: {tsmip_sta}/{station}\n")
                        f.write(f"AI picker -> X\n")
                        f.write(f"Pfile -> Picked!\n")
                        f.write(f"AI picker is False Negative\n")
                        f.write('-'*50)
                        f.write('\n')
                    fn += 1
        except Exception as e:
            with open(savepath, 'a') as f:
                f.write(f"station: {station}\n")
                f.write('No station found!\n')
                f.write('-'*50)
                f.write('\n')
            continue
    avg_diff = time_diff / cnt if cnt != 0 else 0.0

    with open(savepath, 'a') as f:
        f.write(f"AI picker delay {round(avg_diff, 2)} second(s) on average\n") if avg_diff > 0 else f.write(f"AI picker is earlier {round(avg_diff, 2)} second(s) on average\n")
        f.write(f"AI picker -> FP: {fp}, FN: {fn}\n")
        f.write(f"AI picker -> TP: {tp}, TN: {tn}\n")
        f.write('-'*50)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # file path related
    parser.add_argument('--log_dir', type=str, default='./pick_log')
    parser.add_argument('--pfile_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    # event related
    parser.add_argument('--year', type=int)
    parser.add_argument('--month', type=int)
    parser.add_argument('--day', type=int)
    parser.add_argument('--hour', type=int)
    parser.add_argument('--minute', type=int)
    parser.add_argument('--second', type=int)
    parser.add_argument('--lat', type=float)
    parser.add_argument('--lon', type=float)

    # report related
    parser.add_argument('--chunk', type=int)
    parser.add_argument('--time_shift', type=int)
    parser.add_argument('--plot', type=bool, default=False)

    opt = parser.parse_args()
     
    return opt

if __name__ == '__main__':
    opt = parse_args()

    logfiles = glob.glob(opt.log_dir+'/*.log')

    pfiledir = f"./{opt.pfile_dir}/"
    result_path = f"./event_analyze/{opt.output_dir}"
    print('result_path: ', result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # collect the filename as key
    log = {}
    for file in logfiles:
        with open(file, 'r') as f:
            log[file.split('/')[-1].split('.')[0]] = f.readlines()

    print('Loading nsta file...')
    station_list = pd.read_excel("./nsta/Station.xlsx")
    mapping = station_list[['TSMIP_short_code', 'CWBSN_code']]

    chunk = opt.chunk
    sta_list = get_stationlist(chunk)
    sta_list = find_all_cwbsnCode(sta_list, mapping)

    # input data from user
    year, month, day, hour, minute, second = opt.year, opt.month, opt.day, opt.hour, opt.minute, opt.second

    lat, lon = opt.lat, opt.lon
    epicenter = (lon, lat)

    time_shift = opt.time_shift
    search_date = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second) - timedelta(hours=8)

    pfile_list = find_pfilename(opt, pfiledir)
    print('pfile: ', pfile_list)

    compare, p_arr = search(log, epicenter, search_date, time_shift, plot=opt.plot)

    for idx, cur_pfile in enumerate(pfile_list):
        print('Searching...')
        
        pfile = open_P(cur_pfile, sta_list, opt)

        if not pfile == {}:
            find_first_pick(compare, pfile, f"{result_path}/report{idx}", p_arr)
        else:
            print('no pfile')

