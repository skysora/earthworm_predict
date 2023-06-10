
import numpy as np
class Device:
    def __init__(self, station=None, channel=None,network=None,location=None,pick_time=None,scnl=None,station_coord_factor=None):
        self.station = station
        self.channel = channel
        self.network = network
        self.location = location
        self.pick_time = pick_time
        self.scnl = f"{self.station}_{self.channel}_{self.network}_{self.location}"
        self.station_coord_factor = station_coord_factor
        self.lat = 0.0
        self.lon = 0.0
        self.factor = []
        # 0:資料有沒有在station Info 內
        # 1:同個測站重複取值則False
        # 2:座標有沒有在station table 內
        # 3:資料3軸不完整   
        self.status = [False,False,False,False]
        self.depth = None
        self.station_key = None
        self.position = None
        self.scnl_z=None
        self.scnl_n=None
        self.scnl_e=None
    #setter
    def set_coords(self,lat,lon,factor):
        self.lat = lat
        self.lon = lon
        self.factor = np.array(factor)
        
    def set_table_info(self,depth,station_key,position):
        self.depth = depth
        self.station_key = station_key
        self.position = position
        
    def is_data_available(self,scnl_list_position=None,depth_table=None,key_index_table=None,source=None):
        
        # cannot search station Info
        if (not self.factor[0]==-1):
            self.status[0] = True            
        # 同個測站不要重複
        if(self.scnl not in scnl_list_position):
            self.status[1] = True   
        
        # 座標不在table內
        if(f"{self.lon},{self.lat}" in depth_table):
            self.status[2] = True 
        
        # One of 3 channel is not in key_index(i.e. waveform)
        self.scnl_z,self.scnl_n,self.scnl_e = self.get_three_axes_scnl(source)
            
        if not (self.scnl_z not in key_index_table) or (self.scnl_n not in key_index_table) or (self.scnl_e not in key_index_table):
            self.status[3] = True
        
        if(sum(self.status) == 4):
            return True
        else:
            return False
    
    def get_three_axes_scnl(self,source):
        if source == 'Palert' or source == 'CWB' or source == 'TSMIP':
            scnl_z = f"{self.station}_{self.channel[:-1]}Z_{self.network}_{self.location}"
            scnl_n = f"{self.station}_{self.channel[:-1]}N_{self.network}_{self.location}"
            scnl_e = f"{self.station}_{self.channel[:-1]}E_{self.network}_{self.location}"
        else:
            # for tankplayer testing
            if self.channel == 'HHZ': 
                self.channel = ['Ch7', 'Ch8', 'Ch9']
            elif self.channel == 'EHZ': 
                self.channel = ['Ch4', 'Ch5', 'Ch6']
            elif self.channel == 'HLZ': 
                self.channel = ['Ch1', 'Ch2', 'Ch3']
                
            scnl_z = f"{self.station}_{self.channel[0]}_{self.network}_{self.location}"
            scnl_n = f"{self.station}_{self.channel[1]}_{self.network}_{self.location}"
            scnl_e = f"{self.station}_{self.channel[2]}_{self.network}_{self.location}"
            
        return  scnl_z,scnl_n,scnl_e
        
        
        