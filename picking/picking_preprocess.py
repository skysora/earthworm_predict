import numpy as np
import torch
import scipy.signal
import time

def v_to_a(waveform, idx, sampling_rate=100.0):
    waveform = waveform.numpy()
    waveform[idx] = np.gradient(waveform[idx], 1.0/sampling_rate, axis=-1)

    return torch.FloatTensor(waveform)

def filter(waveform, N=5, Wn=[1, 10], btype='bandpass', analog=False, sampling_rate=100.0):
    _filt_args = (N, Wn, btype, analog)

    sos = scipy.signal.butter(*_filt_args, output="sos", fs=sampling_rate)

    # 加長波型，減少因為 filter 導致波型異常現象
    n_append = 100
    n_repeat = 25

    # 用 waveform 前後 100 timesteps 來補足
    tmp_front = waveform[:, :, :n_append].repeat(1, 1, n_repeat)
    tmp_end = waveform[:, :, -n_append:].repeat(1, 1, n_repeat)
   
    # tmp = torch.zeros(waveform.shape[0], waveform.shape[1], n_append)
    toFilter = torch.cat((tmp_front, waveform, tmp_end), dim=-1)
    res = torch.FloatTensor(scipy.signal.sosfilt(sos, toFilter, axis=-1))

    return res[:, :, n_append*n_repeat:-n_append*n_repeat]

def z_score(wave):
    eps = 1e-10

    zeros_idx = torch.where(wave == 0)

    wave = wave - torch.mean(wave, dim=-1, keepdims=True)
    wave = wave / (torch.std(wave, dim=-1, keepdims=True) + eps)

    wave[zeros_idx] = 0

    # padding zeros with neighborhood'd values
    wave, nonzero_flag = padding(wave)

    return wave, nonzero_flag

def calc_feats(waveforms):
    CharFuncFilt = 3
    rawDataFilt = 0.939
    small_float = 1.0e-10
    STA_W = 0.6
    LTA_W = 0.015
    
    # filter
    result = torch.empty((waveforms.shape))
    data = torch.zeros((waveforms.shape[0], 3))

    for i in range(waveforms.shape[2]):
        if i == 0:
            data = data * rawDataFilt + waveforms[:, :, i] + small_float
        else:
            data = (
                data * rawDataFilt
                + (waveforms[:, :, i] - waveforms[:, :, i - 1])
                + small_float
            )

        result[:, :, i] = data

    wave_square = torch.square(result)

    # characteristic_diff
    diff = torch.empty((result.shape))

    for i in range(result.shape[2]):
        if i == 0:
            diff[:, :, i] = result[:, :, 0]
        else:
            diff[:, :, i] = result[:, :, i] - result[:, :, i - 1]

    diff_square = torch.square(diff)

    # characteristic's output vector
    wave_characteristic = torch.add(
        wave_square, torch.mul(diff_square, CharFuncFilt)
    )

    # sta
    sta = torch.zeros((waveforms.shape[0], 3))
    wave_sta = torch.empty((waveforms.shape))

    # Compute esta, the short-term average of edat
    for i in range(waveforms.shape[2]):
        sta += STA_W * (waveforms[:, :, i] - sta)

        # sta's output vector
        wave_sta[:, :, i] = sta

    # lta
    lta = torch.zeros((waveforms.shape[0], 3))
    wave_lta = torch.empty((waveforms.shape))

    # Compute esta, the short-term average of edat
    for i in range(waveforms.shape[2]):
        lta += LTA_W * (waveforms[:, :, i] - lta)

        # lta's output vector
        wave_lta[:, :, i] = lta
  
    # concatenate 12-dim vector as output
    waveforms = torch.cat(
        (waveforms, wave_characteristic, wave_sta, wave_lta), dim=1
    )


    return waveforms

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def padding_with_case(wave, pad_case, pad_idx, n_neighbor=100):
    wavelength = 3000
    
    if pad_case == 1:
        toPad_value = wave[pad_idx[-1]:pad_idx[-1]+n_neighbor] if pad_idx[-1]+n_neighbor < wavelength else wave[pad_idx[-1]:]
        n_neighbor = toPad_value.shape[0]
        toPad_length = pad_idx[-1] - pad_idx[0]

        if n_neighbor == 0:
            return torch.from_numpy(wave)
        
        if not toPad_length <= 10:
            # print(f"1 toPad_value: {toPad_value.shape}, n-neighbor: {n_neighbor}, toPad_length: {toPad_length}")
            toPad = np.tile(toPad_value, toPad_length // n_neighbor)
            remaining = toPad_length % n_neighbor
            # print(f"toPad: {toPad.shape}, remaining: {remaining}")
            if remaining > 0:
                toPad = np.concatenate((toPad, toPad_value[:remaining]))

            wave[pad_idx[0]:pad_idx[-1]] = toPad

    elif pad_case == 2:
        toPad_value = wave[pad_idx[0]-n_neighbor:pad_idx[0]] if pad_idx[0]-n_neighbor >= 0 else wave[:pad_idx[0]]
        n_neighbor = toPad_value.shape[0]        
        toPad_length = pad_idx[-1] - pad_idx[0]

        if n_neighbor == 0:
            return torch.from_numpy(wave)
        
        if not toPad_length <= 10:
            toPad = np.tile(toPad_value, toPad_length // n_neighbor)
            remaining = toPad_length % n_neighbor
            if remaining > 0:
                toPad = np.concatenate((toPad, toPad_value[:remaining]))
    
            wave[pad_idx[0]:pad_idx[-1]] = toPad

    else:
        toPad_value = wave[pad_idx[0]-n_neighbor:pad_idx[0]] if pad_idx[0]-n_neighbor >= 0 else wave[:pad_idx[0]]
        n_neighbor = toPad_value.shape[0]
        toPad_length = pad_idx[-1] - pad_idx[0]

        if n_neighbor == 0:
            return torch.from_numpy(wave)
        
        if not toPad_length <= 10:
            toPad = np.tile(toPad_value, toPad_length // n_neighbor)
            remaining = toPad_length % n_neighbor
            
            if remaining > 0:
                toPad = np.concatenate((toPad, toPad_value[:remaining]))

            wave[pad_idx[0]:pad_idx[-1]] = toPad

    return torch.from_numpy(wave)

def padding(a):
    '''
    Three types of padding:
    1) Zeros is at the beginning of the waveform.
    2) Zeros is at the end of the waveform.
    3) Zeros is at the middle of the waveform.
    Note that multiple cases may occur in a single waveform.
    '''
    nonzero_flag = [True for _ in range(a.shape[0])]
    for batch in range(a.shape[0]):
        try:
            wave = a[batch]
            backup_wave = a[batch].clone()

            # check waveform full of zeros
            if torch.all(wave == 0):
                nonzero_flag[batch] = False
                continue

            # finding zero values alone Z, N, E axis
            zeros = [[zero_runs(wave[i].numpy())] for i in range(wave.shape[0])]
            
            # padding follows the order: Z -> N -> E
            batch_pad = False
            for i in range(len(zeros)):
                # There is no zero in the trace
                if zeros[i][0].shape[0] == 0:
                    continue

                for row, j in enumerate(zeros[i][0]):
                    isPad = False

                    # check first row contain "0" or not, if not, then padding_case 1 is not satisfied.
                    if j[0] == 0:
                        # padding case 1
                        wave[i] = padding_with_case(wave[i].numpy(), 1, j)
                        isPad = True
                        batch_pad = True

                    # check the last row contain "wavelength-1" or not, if not, then padding_case 3 is not satisfied.
                    if j[-1] == wave.shape[-1]:
                        # padding case 3
                        wave[i] = padding_with_case(wave[i].numpy(), 3, j)
                        isPad = True
                        batch_pad = True

                    # check the middle rows
                    if not isPad:
                        wave[i] = padding_with_case(wave[i].numpy(), 2, j)
                        batch_pad = True

                a[batch] = wave
            
            if batch_pad:
                nonzero_flag[batch] = False
        except Exception as e:
            # print(e)
            a[batch] = backup_wave
            
    return a, nonzero_flag

