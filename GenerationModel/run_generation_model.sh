#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python demo_ED_finetune.py --data small

: <<'END'
demo_ED_finetune.py : 使用EmpatheticDialogues對DialoGPT進行微調(參考原DialoGPT GitHub: demo.py)
參數:
    data 原demo.py用來指定驗證集之參數，此版本已將訓練集與測試集於程式中指定為EmpatheticDialogues，因此可忽略其實質效果
檔案參考來源為
microsoft/DialoGPT : https://github.com/microsoft/DialoGPT
將其使用之資料集改為EmpatheticDialogues作微調訓練
此python檔案會呼叫LSP_train.py進行訓練，主要參數依照原始設定，額外改動部分為: 預訓練模型size、資料集、batch_size(防止out of memory)

END