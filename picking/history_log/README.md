
## 自動化比對 log 檔案與氣象局 pfile 的 p_arrival_time
  
  * 讀取氣象局 metadata，自動將 TSMIP, CWBSN station code 對應上
  * 比對有在 pfile or log file 出現的測站 p-wave arrival time
  * 自動繪製每秒鐘震央與全台測站 picking 情形

  * Arguments:
      * `year`, `month`, `day`, `hour`, `minute`, `second`: 發震時間相關參數
      * `lon`, `lat`: 震央座標
      * `pfile_dir`: pfile 所在的資料夾
      * `output_dir`: 報告產生所在的資料夾
      * `time_shift`: 從發震時間算起，要往後繪製幾秒的全台測站 picking 圖
      * `log_dir`: 欲比對的 log file 所在的資料夾

```python
python check_log.py --chunk <chunk of log file> \
--year <event_year> --month <event_month> --day <event_day> --minute <event_minute> --second <event_second> \
--lon <epicenter_lon> --lat <epicenter_lat> \
--pfile_dir <path to directory where pfile located> --output_dir <path to report> \
--time_shift <how many seconds to plot the picked station> \
--log_dir <path to directory where log file located> 
```
