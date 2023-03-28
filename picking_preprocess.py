import numpy as np
import torch
import scipy.signal

def v_to_a(waveform, idx, sampling_rate=100.0):
    waveform = waveform.numpy()
    waveform[idx] = np.gradient(waveform[idx], 1.0/sampling_rate, axis=-1)

    return torch.FloatTensor(waveform)

def filter(waveform, N=5, Wn=[1, 45], btype='bandpass', analog=False, sampling_rate=100.0):
    _filt_args = (N, Wn, btype, analog)

    sos = scipy.signal.butter(*_filt_args, output="sos", fs=sampling_rate)

    return torch.FloatTensor(scipy.signal.sosfilt(sos, waveform, axis=-1))

def z_score(wave):
    eps = 1e-10

    # wave = wave - np.mean(wave, axis=-1, keepdims=True)
    # wave = wave / (np.std(wave, -1, keepdims=True) + eps)

    wave = wave - torch.mean(wave, dim=-1, keepdims=True)
    wave = wave / (torch.std(wave, dim=-1, keepdims=True) + eps)

    return wave

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
