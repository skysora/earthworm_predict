# Deep-learning Picker

## Execute
```
CUDA_VISIBLE_DEVICES=gpu_id or "" python EEW.py
```

## Model
- Encoder: Conformer
- Decoder: Cross-attention + Transformer

## Checkpoint path
- https://drive.google.com/file/d/1DAtnzUS244aQmYRwpYNJ_ZYXfRbAuG9j/view?usp=share_link

## Introduction
- 結合 `WaveSaver` 和 `PickHandler`.
- `Picker`: 串接 WAVE_RING 的波型與資訊, 輸入進模型做 picking 預測, 再把有 pick 到的測站資訊與 p 波到時寫進 PICK_RING.
- 只取用 `waveform_buffer` 中間 3000 個 samples 的波型來做預測.

## BUGs
- 如果在 `PickHandler` 中一直出現 XXX_XXX_XXX_XX one of 3 channel missing 的錯誤, 就到 `Picker` 的 297-310 行程式碼, 把後方註解有 `for Ch1-9` 的註解掉, 再把後方註解有 `for channel = XXZ` 的打開.

  - 因為 tankplayer 回放 afile 時收到的封包, channel 欄位都是 Ch1-9, 而 CWB 與 Palert 都是 HLZ, EHZ, HHZ 這類型
