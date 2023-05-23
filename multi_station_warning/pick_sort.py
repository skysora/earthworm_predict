#計算pick順序
pick_now = time.time()
for pick_info in wait_list:

    pick_info = pick_info.split(' ')
    station = pick_info[0]
    channel = pick_info[1]
    network = pick_info[2]
    location = pick_info[3]
    scnl = f"{station}_{channel}_{network}_{location}"
    station_coord_factor = get_coord_factor(np.array([scnl]), stationInfo)[0]
    # cannot search station Info
    if (station_coord_factor[1]==-1):
        print(f"{scnl} not find in stationInfo")
        wait_list.pop(station_index)
        if(len(wait_list)>0):
            continue
        else:
            break
    depth = dataDepth[f"{station_coord_factor[1]},{station_coord_factor[0]}"]
    station_key = f"{station_coord_factor[1]},{station_coord_factor[0]},{depth}"
    if(station_key in stations_table.keys()):
        position = stations_table[station_key]
        pick_index = int((float(pick_now)-float(pick_info[10]))*100)
        if(position in wait_list_position):
            wait_list_pick_index[wait_list_position.index(position)] = pick_index
            continue
        wait_list_pick_index.append(pick_index)
        wait_list_position.append(position)
        ok_wait_list.append(pick_info)
    else:
        print(f"{station_key} not in station_table")
        pass
        
    station_index+=1
    
if(len(wait_list_pick_index) > 0): 
    wait_list_sort_index = np.argsort(np.array(wait_list_pick_index))
    first_station_index = wait_list_pick_index[wait_list_sort_index[0]]
    first_station_position = wait_list_position[wait_list_sort_index[0]]
