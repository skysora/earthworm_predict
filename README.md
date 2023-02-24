# Pyearthworm-predict-pga25

## 簡介
- 一個串接Earthworm波形，並把pick到的測站的波形餵給模型預測結果的程式
- 由於CWB的波形是一軸一個封包，且順序未定、可能被其他測站的其他軸插隊、單個封包的波形長度不固定，因此需要有個程式將其bind在一起再一同餵給模型當輸入

## cnn_cls_predict_pyearthworm_pga25
- 大部分參數於 `.env` 中修改即可
- 分為兩大部分，`WaveSaver` 和 `PickHandler`
### WaveSaver
- 接收波形並存入 `waveform_buffer`
- `waveform_buffer` 為一個 shape 為 (各測站各軸數量, 6000) 的 nparray，每個 element 為長度 6000 (60秒) 的 buffer(nparray)，範圍為目前時間 +-30 秒，每 5 秒會更新一次往右 shift，並為新加入的 500 個sample (5秒) 填入 0
- `key_index` 為一個用來存各測站各軸的 index 的 dict，用來對照 `waveform_buffer` 的位置用的(因為 nparray 不能用 key 搜尋)，當收到一個波形封包時，會先解析該封包的 SCNL，並去 `key_index` 尋找他在 `waveform_buffer` 中的 index 位置，再將波形資訊存入其中
- `waveform_buffer_start_time` 是一個用來記錄 `waveform_buffer` 中第 0 個位置的 epoch 時間，如此在收到波形封包時，便可根據封包的起始時間與該參數資訊填入 `waveform_buffer` 對應的時間點，就可以確保其存入的波形位置一定是那個時間點的

### PickHandler
- 監聽 PICK_RING，如果有看到測站被 pick 到，會先放進 `wait_list` 中等待給模型處理
- 每次會去抓 `wait_list` 的第 0 個元素，接著根據該測站資訊去 `waveform_buffer` 找對應的波形
- 根據 pick 到的測站資訊中的 `pstime` 去 `waveform_buffer` 抓對應秒數後的波形資料
- 找到後會去查看任一軸有沒有超過欲預測的秒數一半以上都是 0 (如: 300 的模型超過 150 個 sample 都是 0)，則會把他踢出 `wait_list`
- 最後將波形餵給模型預測，若結果為 yes 就會把它放進另一個 OUTPUT_RING 中，等待給 `send_tcpy.py` 處理

## send_tcpd
- 大部分參數於 `.env` 中修改即可
- 監聽 OUTPUT_RING，若有測站被放進去就會放進 `existStation`，存活時間為 `p_survive` 秒
- 若 `existStation` 數量超過 `sta_count` 則會去看這些測站間的距離有沒有在 `trig_dis` 範圍內
- 若有的話就會發一個地震報告到 RING 中，發完後會休息 `p_survive` 秒再去 `existStation` 看需不需要再發報# earthworm_predict
