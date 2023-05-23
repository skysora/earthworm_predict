# Real-time Picker

## Usage
```
CUDA_VISIBLE_DEVICES=<gpu_id or ""> python EEW.py
```
---
## Model
- Encoder: Conformer
- Decoder: Cross-attention + Transformer
---
## Checkpoint path
- Conformer_crossattn_prevTaiwan: https://drive.google.com/file/d/1DAtnzUS244aQmYRwpYNJ_ZYXfRbAuG9j/view?usp=share_link
- Conformer_crossattn_Taiwan: https://drive.google.com/file/d/1ISOEN8KPlVHEPYEjOkIzYvkEYWk33RZn/view?usp=sharing
- Conformer_TSMIP_HL: https://drive.google.com/file/d/1gAzVA9pNWoi8Aa6DivsCSZTqDuazfs09/view?usp=sharing
- Conformer_EEW_other: https://drive.google.com/file/d/1K2UAndV7pMefFC3gLVP5oipW7_6uLE4n/view?usp=share_link
- EQTransformer_Taiwan: https://drive.google.com/file/d/13A1oZ-EvlfZN7noxpH_wrOJYg8b4yx1b/view?usp=share_link

---
## Introduction
- 支援三種即時資料模式：`CWBSN tank player`, `TSMIP real time`, and `Palert real time`
- WaveSaver 預先切好相對應模式的測站區域，並只存取該區域測站的波形
- 每秒做預測後，若有測站被 picked，則產生相對應的 picking message，並寫入 PICK_RING
- 若超過門檻，則使用 line notify 發報
- 除了發報情況以外，每秒都會有一定機率產生預測圖形，並藉由 line notify 傳送
- 只要有 pick 到，都會寫入 picking_chunk?.log；若有發報則同步寫入 notify_chunk?.log
- Buffer 長度通常設為 **6000**，第 3000 sample 為目前時間點
---
## Basic configs
* 在氣象局裝的版本，接收 TSMIP 即時測站，共 553 個測站 (`TSMIP real time`)

* **PyEarthworm 相關參數**

| Parameter | Value |
|:-------:|:-------:|
| PYEW_MODULE_ID | 190 |
| PYEW_INST_ID | 52 |
| WAVE_RING_ID | 1034 |
| PICK_RING_ID | 1005 |
| PICK_MSG_TYPE | 150 |
| OUTPUT_MSG_TYPE | 98 | 
| REPORT_MSG_TYPE | 150 |

* **三種模式與其對應檔案、切割方式**
  - [TSMIP 切割方式](https://github.com/earthquake-NLPLAB/pyearthworm-predict-pga25/blob/master/picking/TSMIP_region.md)
  
| Mode| Related file |
|:-------:|:-------:|
| CWBSN tank player | ./nsta/nsta24.dat |
| TSMIP real time | ./nsta/sta_csmt_Z |
| Palert real time | ./nsta/palertlist.csv | 

---
## Picking configs
| Parameter | Meaning | Example |
|:-------:|:-------:|:-------:|
| CHUNK | 測站劃分區域後，第幾個區域 | 0 |
| SOURCE | 即時資料模式 | TSMIP |
| STORE_LENGTH | Buffer 長度 | 6000 |
| N_PREDICTION_STATION | 這個 chunk 共有幾個測站 | 74 |
| THRESHOLD_TYPE | Picking rule (type) | continue |
| THRESHOLD_PROB | Picking rule (probability) | 0.65 |
| THRESHOLD_TRIGGER | Picking rule (sample) | 5 |
| THRESHOLD_KM | 方圓 x 公里測站 picking threshold | 20 |
| THRESHOLD_NEIGHBOR | 鄰近 x 個測站 picking threshold | 2 |
| NEAREST_STATION | 鄰近 THRESHOLD_NEIGHBOR 個測站中，有 x 個 picked 才算 | 3 |
| TABLE_OPTION | picking postprocess 方式 | nearest |
| REPORT_NUM_OF_TRIGGER | 幾個測站 picked 才要發報 | 40 |
| DELETE_PICKINGLOG_DAY | 刪除幾天前的 picking log 檔 | 3 |
| DELETE_NOTIFYLOG_DAY | 刪除幾天前的 notify log 檔 | 30 |
| KEEP_WAVE_DAY | 每天最多保存多少波型 | 100 |


---
 ## 氣象局檢查雜訊資料 SOP
 * **工作目錄**: /home/rtd/Earthworm/conformer_picking/
 (以下皆於工作目錄底下作業)
 
 0. **Setup environment**:
 ```shell
 cd /home/rtd/Earthworm/conformer_picking/

 conda activate conformer_picking
 ```
 
 1. **Download**: 先至 ./plot 資料夾將 "trigger" 與 "checked" 以外的事件資料夾下載
 ```
 Folder name example: 2023-03-20 04:11:49.370000 -> 系統 buffer 起始時間
 ```
 
 2. **Check**: 每個事件底下包含四個子資料夾，人工檢查圖檔後請依功能放入對應資料夾:
 
    * **unchecked**: 儲存尚未檢查過的波形圖 (png 檔)
    
    * **noise**: 將人工檢查完**確定為雜訊**的圖檔放入此資料夾
    
    * **unavailable**: 將人工檢查完**波形有異常**的圖檔放入此資料夾
    
    * **seismic**: 將人工檢查完**確定為地震訊號**的圖檔放入此資料夾
    
 3. **Upload the results**: 將檢查完的整個事件資料夾直接上傳至 ./plot/checked
 
 4. **Upload to Google Drive**: 請至工作目錄底下執行以下指令:
 ```python
 python upload_noise.py
 ```
 
 ### File structures example
 
    ├── ./plot                            # 存放所有圖檔的根目錄
    │   ├── trigger                       # 儲存發報的台灣地圖 (不用管)
    │   ├── checked                       # 儲存人工檢查完的事件資料夾 (步驟三將事件資料夾上傳至此)
    │   ├── 2023-03-20 04:11:49.370000    # 單一事件資料夾 (步驟一直接下載這個資料夾)
    │       ├── unchecked                 # 儲存尚未檢查過的波形圖 (png 檔)
    │       ├── noise                     # 人工檢查完確定為雜訊的圖 (png 檔)
    │       ├── unavailable            # 人工檢查完可能為雜訊的圖 (png 檔)
    │       ├── seismic                   # 人工檢查完確定為地震訊號的圖 (png 檔)
    
