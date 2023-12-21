#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_valence_regerssion.py --text_file data/SST-5/split/train_sentence.txt --condition_file data/SST-5/split/train_valence.txt --bert_model bert-base-uncased --output_dir bert_epoch20_sst5 --label_num 1 --max_seq_length 80 --do_train --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 20
CUDA_VISIBLE_DEVICES=0 python3 exportValenceProfile.py --text_file data/SST-5/split/train_sentence.txt --bert_model bert-base-uncased  --label_num 1 --max_seq_length 80 --eval_batch_size 8 --load_config bert_epoch20_sst5/bert_config/32220config.json --load_model bert_epoch20_sst5/bert_model/32220pytorch_model.bin --do_eval --object_type usr
CUDA_VISIBLE_DEVICES=0 python3 exportValenceProfile.py --text_file data/SST-5/split/test_sentence.txt --bert_model bert-base-uncased  --label_num 1 --max_seq_length 80 --eval_batch_size 8 --load_config bert_epoch20_sst5/bert_config/32220config.json --load_model bert_epoch20_sst5/bert_model/32220pytorch_model.bin --do_eval --object_type usr

CUDA_VISIBLE_DEVICES=0 python3 run_valence_regerssion_ftSST.py --text_file data/ED_situ/train_situation.txt --condition_file data/ED_situ/train_situation_valence.txt --bert_model bert-base-uncased --output_dir bert_epoch20_ftsitua --label_num 1 --max_seq_length 80 --do_train --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 20
CUDA_VISIBLE_DEVICES=0 python3 exportValenceProfile.py --text_file data/ED_situ/train_situation.txt --bert_model bert-base-uncased  --label_num 1 --max_seq_length 80 --eval_batch_size 8 --load_config bert_epoch20_ftsitua/bert_config/32220config.json --load_model bert_epoch20_ftsitua/bert_model/32220pytorch_model.bin --do_eval --object_type usr
CUDA_VISIBLE_DEVICES=0 python3 exportValenceProfile.py --text_file data/ED_situ/test_situation.txt --bert_model bert-base-uncased  --label_num 1 --max_seq_length 80 --eval_batch_size 8 --load_config bert_epoch20_ftsitua/bert_config/32220config.json --load_model bert_epoch20_ftsitua/bert_model/32220pytorch_model.bin --do_eval --object_type usr

: <<'END'
run_valence_regerssion.py : 使用SST-5資料訓練emotional valence回歸模型(用使用者語句偵測使用者該句話的emotional valence)
參數:
    text_file 語料
    condition_file 相對應的emotional valence
    bert_model bert預訓練模型
    output_dir 輸出位置 
    label_num 標籤數
    max_seq_length 句子最長字元數 
    do_train 訓練
    num_train_epochs 訓練epoch數

run_valence_regerssion_ftSST.py : 使用EmpatheticDialogues進行微調emotional valence回歸模型(用使用者語句偵測使用者該句話的emotional valence, 在code中將預訓練模型位置設為SST-5訓練過的模型)
參數:
    text_file 語料
    condition_file 相對應的emotional valence
    bert_model bert預訓練模型
    output_dir 輸出位置 
    label_num 標籤數
    max_seq_length 句子最長字元數 
    do_train 訓練
    num_train_epochs 訓練epoch數

exportEmoProfile_empathy.py : 使用訓練好的情緒分類器進行標記
參數:
    text_file 要被標記情緒的語料
    bert_model bert預訓練模型
    load_model 已訓練好的情緒分類模型位置
標記結果會放在load_model此位置下的 /total/usr資料夾
valence_pred__xxx.txt 是emotion valence偵測結果

emotion_transfer_valence_table.txt : 32種對上其相應的emotional valence大小
emotion_transfer_valence.py :將情緒轉換成其相對的emotional valence標籤
emotional valence都轉好了 所以這個應該不太需要用到

END