import time
import numpy as np
import torch
import torch.nn.functional as F
from model_resnet import ResNet18
from datetime import datetime, timedelta
from collections import deque
import threading
import PyEW
from dotenv import dotenv_values



class model_cnn():
    def __init__(self, model=None, param_path=None):
        self.model = model
        checkpoint = torch.load(param_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def output(self, input):
        input = np.expand_dims(input,axis=0)
        input = torch.from_numpy(input).float()
        with torch.no_grad():
            probs = torch.squeeze(F.softmax(self.model(input), dim=1)).numpy()

        return probs

    def output_max(self, input):
        input = np.expand_dims(input,axis=0)
        input = torch.from_numpy(input).float()
        with torch.no_grad():
            output = np.argmax(torch.squeeze(F.softmax(self.model(input), dim=1)).numpy())

        return output

# to save wave form from WAVE_RING
class WaveSaver(threading.Thread):
    def __init__(self, thread_name):
        super(WaveSaver, self).__init__(name=thread_name)
    
    def run(self):
        global waveform_buffer
        global MyModule
        global key_index
        global nowtime
        global waveform_buffer_start_time

        key_cnt = 0 # a counter for accumulating key's count

        while True:
            # get raw waveform from WAVE_RING
            wave = MyModule.get_wave(0)

            # keep getting wave until the wave isn't empty
            if wave == {}:
                wave = MyModule.get_wave(0)
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
                key_index[scnl] = key_cnt
                # initialize the scnl's waveform with shape of (1,6000) and fill it with 0
                waveform_buffer = np.append(waveform_buffer, np.zeros((1,6000)), axis = 0)
                key_cnt += 1

            # find input time's index to save it into waveform accouding this index
            startIndex = int(startt*100) - waveform_buffer_start_time
            if startIndex < 0:
                # the wave is too old to save into buffer
                wave = MyModule.get_wave(0)
                continue
            
            # save wave into waveform from starttIndex to its wave length
            try:
                waveform_buffer[key_index[scnl]][startIndex:startIndex+nsamp] = wave['data']
            except:
                print(f"{scnl} can't assign wave data into waveform")

            # move the time window of timeIndex and waveform every 5 seconds
            if int(time.time()*100) - nowtime >= 500:
                # print(f"{time.time()} updating waveform and timeIndex for {nowtime}")
                waveform_buffer_start_time += 500
                waveform_buffer[:, 0:5500] = waveform_buffer[:, 500:6000]
                # the updated waveform is fill in with 0
                waveform_buffer[:, 5500:6000] = np.zeros((waveform_buffer.shape[0],500))
                nowtime += 500

            

# to handle data when there is a new station be picked
class PickHandler(threading.Thread):
    def __init__(self, thread_name):
        super(PickHandler, self).__init__(name=thread_name)

    def run(self):
        global waveform_buffer
        global MyModule
        global mode_cnn
        global key_index
        global waveform_buffer_start_time
        global env_config

        used_second = 3
        sample = 100
        length = used_second * sample
        time.sleep(used_second)

        PICK_MSG_TYPE = int(env_config["PICK_MSG_TYPE"])
        OUTPUT_MSG_TYPE = int(env_config["OUTPUT_MSG_TYPE"])

        # read factor from sta_CWB24_Z, because each station has different factor(to transfer count to gal) in CWB
        with open(env_config["STA_FACTOR_PATH"],"r") as fp:
            data=fp.readlines()
        factor={}
        for i in data:
            tmp=i.split()
            sta = tmp[0]
            chn = tmp[1]
            net = tmp[2]
            loc = tmp[3]
            fac = float(tmp[-2])
            tmp_scnl = f"{sta}_{chn}_{net}_{loc}"
            factor[tmp_scnl]=fac


        # a waiting list for model to process if multiple stations comes into PICK_RING simultaneously
        wait_list = deque()

        # listen PICK_RING
        while True:
            log_msg = "============================"

            # get picked station in PICK_RING
            pick_msg = MyModule.get_bytes(1, PICK_MSG_TYPE)

            # if there's no data and waiting list is empty
            if pick_msg == (0, 0) and len(wait_list) == 0:
                continue

            # if get picked station, then get its info and add to waiting list
            if pick_msg != (0, 0):
                pick_str = pick_msg[2][:pick_msg[1]].decode("utf-8")
                #print(time.time(), pick_str)

                # the quality of wave should not greater than 2
                if int(pick_str[-5]) > 2:
                    continue

                # the using time period for picking module should be 3 seconds
                if int(pick_str[-1]) !=3:
                    continue
            
                log_msg += "\n[" + str(time.time()) + "] " + pick_str
                wait_list.append(pick_str)

            # get the first one data in waiting list
            pick_info = wait_list[0].split()
            station = pick_info[0]
            channel = pick_info[1]
            network = pick_info[2]
            location = pick_info[3]
            scnl = f"{station}_{channel}_{network}_{location}"
            scnl_z = f"{station}_{channel[:2]}Z_{network}_{location}"
            scnl_n = f"{station}_{channel[:2]}N_{network}_{location}"
            scnl_e = f"{station}_{channel[:2]}E_{network}_{location}"
            # print(scnl)
            pstime = float(pick_info[-4])

            # sometimes the ptime of picked station would be a large number
            if pstime - time.time() > 5:
                print("pstime error")
                wait_list.popleft()
                continue

            # One of 3 channel is not in key_index(i.e. waveform)
            if (scnl_z not in key_index) or (scnl_n not in key_index) or (scnl_e not in key_index):
                print(f"{scnl} one of 3 channel missing")
                wait_list.popleft()
                continue

            # check if sta factor has data of the picked station
            if scnl_z not in factor:
                wait_list.popleft()
                continue

            # get the index of z,n,e in wavoform
            z_waveform_index = key_index[scnl_z]
            n_waveform_index = key_index[scnl_n]
            e_waveform_index = key_index[scnl_e]

            # find ptime's time index in waveform
            pstime_index = int(pstime*100) - waveform_buffer_start_time
            if pstime_index < 0:
                # print(f"time index not found\npstime: {pstime} }")
                wait_list.popleft()
                continue

            # if there's no more than {used_second} in waveform to predict
            if pstime_index + length > 6000:
                print(f"less than {used_second} seconds")
                wait_list.popleft()
                continue

            # get waveform of z,n,e starting from ptime
            hlz = waveform_buffer[z_waveform_index][pstime_index:pstime_index+length]
            hln = waveform_buffer[n_waveform_index][pstime_index:pstime_index+length]
            hle = waveform_buffer[e_waveform_index][pstime_index:pstime_index+length]
            
            # check if there are more than half of length are zeros in one of 3 channels
            z_zerocount = np.count_nonzero(hlz==0)
            if z_zerocount >= length/2:
                print(f"{scnl_z} has more than {length/2} zeros")
                wait_list.popleft()
                continue

            n_zerocount = np.count_nonzero(hln==0)
            if n_zerocount >= length/2:
                print(f"{scnl_n} has more than {length/2} zeros")
                wait_list.popleft()
                continue

            e_zerocount = np.count_nonzero(hle==0)
            if e_zerocount >= length/2:
                print(f"{scnl_e} has more than {length/2} zeros")
                wait_list.popleft()
                continue

            # substract mean
            hlz = hlz - np.mean(waveform_buffer[z_waveform_index][:pstime_index+length])
            hln = hln - np.mean(waveform_buffer[n_waveform_index][:pstime_index+length])
            hle = hle - np.mean(waveform_buffer[e_waveform_index][:pstime_index+length])

            inp = np.array([hln[:], hle[:], hlz[:]])/factor[scnl_z]
            startt_local = datetime.utcfromtimestamp(pstime) + timedelta(hours=8)
            endt_local = datetime.utcfromtimestamp(pstime + used_second) + timedelta(hours=8)

            # print result
            msg = '\n============================'
            msg += ('\nstation: ' + scnl)
            msg += ('\nstart: ' + startt_local.strftime('%Y-%m-%d %H:%M:%S.%f'))
            msg += ('\nend: ' + endt_local.strftime('%Y-%m-%d %H:%M:%S.%f'))
            print('start predict:',time.time())
            log_msg += "\n[" + str(time.time()) + "] start predict"
            prob = mode_cnn.output(inp)
            print('end predict:',time.time())
            log_msg += "\n[" + str(time.time()) + "] end predict"
            msg += ("\nProbability: " + str(prob))
            msg += ("\nPGA>25: ")
            if prob[0] <= prob[1]:
                msg += "yes"
                # put result to RING
                MyModule.put_msg(2, OUTPUT_MSG_TYPE, wait_list[0])
            else:
                msg += "no"
            msg += '\n============================'
            print(msg)
            log_msg += msg

            # write log to file
            with open("pick_info_pga25_log.txt","a") as pif:
                pif.write(log_msg)
                pif.close()

            wait_list.popleft()
            print(wait_list)

try:
    env_config = dotenv_values(".env")

    # to save all raw wave form data, which is the numpy array, and the shape is (station numbur*channel, 3000)
    waveform_buffer = np.empty(shape=(0,6000))
    
    
    key_index = {} # a dict for checking scnl's index of waveform

    # a deque from time-3000 to time for time index
    nowtime = int(time.time()*100)
    waveform_buffer_start_time = nowtime - 3000

    # connect to earthworm, add WAVE_RING and PICK_RING and an OUTPUT_RING
    MyModule = PyEW.EWModule(int(env_config["WAVE_RING_ID"]), int(env_config["PYEW_MODULE_ID"]), int(env_config["PYEW_INST_ID"]), 30.0, False)
    MyModule.add_ring(int(env_config["WAVE_RING_ID"])) # WAVE_RING
    MyModule.add_ring(int(env_config["PICK_RING_ID"])) # PICK_RING
    MyModule.add_ring(int(env_config["OUTPUT_RING_ID"])) # OUTPUT_RING

    # initialize and load model
    mode_cnn = model_cnn(model=ResNet18(), param_path="300_cwb+tsmip_resnet18_dropout0.2.ckpt")
    
    wave_count = 0
    # flush PICK_RING
    while MyModule.get_bytes(1, int(env_config["PICK_MSG_TYPE"])) != (0, 0):
        wave_count += 1
        continue
    print("PICK_RING flushed with " + str(wave_count) + " waves flushed")

    wave_count = 0
    # flush WAVE_RING
    while MyModule.get_wave(0) != {}:
        wave_count += 1
        continue
    print("WAVE_RING flushed with " + str(wave_count) + " waves flushed")

    wave_saver = WaveSaver('waveServer')
    wave_saver.start()

    pick_handler = PickHandler('pickHandler')
    pick_handler.start()
except KeyboardInterrupt:
    MyModule.goodbye()
