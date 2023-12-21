#!/bin/bash

# original dialogpt (with finetune)
CUDA_VISIBLE_DEVICES=5 python3 dialogGPT_generation.py --out_dir fortest --for_test_run for_test_run --finetune_generation_model ./models/GenModel/ED-pretrain-step-10000.pkl

# perturb
CUDA_VISIBLE_DEVICES=5 python3 ED_pplm_sen_all.py --out_dir fortest --for_test_run for_test_run --length 80 --num_samples 1 --sample -M microsoft/DialoGPT-medium --verbosity quiet --finetune_generation_model ./models/GenModel/ED-pretrain-step-10000.pkl --emotional_valence_config ./models/EmoVal/bert_config/32220config.json --emotional_valence_model ./models/EmoVal/bert_model/32220pytorch_model.bin --information_relevance_config ./models/InfRel/medium/config.json --information_relevance_model ./models/InfRel/medium/ED-reverseV2-step-10000.pkl
CUDA_VISIBLE_DEVICES=5 python3 ED_pplm_sen_rel.py --out_dir fortest --for_test_run for_test_run --length 80 --num_samples 1 --sample -M microsoft/DialoGPT-medium --verbosity quiet --finetune_generation_model ./models/GenModel/ED-pretrain-step-10000.pkl --emotional_valence_config ./models/EmoVal/bert_config/32220config.json --emotional_valence_model ./models/EmoVal/bert_model/32220pytorch_model.bin --information_relevance_config ./models/InfRel/medium/config.json --information_relevance_model ./models/InfRel/medium/ED-reverseV2-step-10000.pkl
CUDA_VISIBLE_DEVICES=5 python3 ED_pplm_sen_val.py --out_dir fortest --for_test_run for_test_run --length 80 --num_samples 1 --sample -M microsoft/DialoGPT-medium --verbosity quiet --finetune_generation_model ./models/GenModel/ED-pretrain-step-10000.pkl --emotional_valence_config ./models/EmoVal/bert_config/32220config.json --emotional_valence_model ./models/EmoVal/bert_model/32220pytorch_model.bin --information_relevance_config ./models/InfRel/medium/config.json --information_relevance_model ./models/InfRel/medium/ED-reverseV2-step-10000.pkl
CUDA_VISIBLE_DEVICES=5 python3 ED_pplm_sen_kl.py --out_dir fortest --for_test_run for_test_run --length 80 --num_samples 1 --sample -M microsoft/DialoGPT-medium --verbosity quiet --finetune_generation_model ./models/GenModel/ED-pretrain-step-10000.pkl --emotional_valence_config ./models/EmoVal/bert_config/32220config.json --emotional_valence_model ./models/EmoVal/bert_model/32220pytorch_model.bin --information_relevance_config ./models/InfRel/medium/config.json --information_relevance_model ./models/InfRel/medium/ED-reverseV2-step-10000.pkl

# # bleu
python3 bleu.py -pred_file ./output/DialoGPT.txt -label_file ./data/test_system_utter.txt
python3 bleu.py -pred_file ./output/perturb_val.txt -label_file ./data/test_system_utter.txt
python3 bleu.py -pred_file ./output/perturb_rel.txt -label_file ./data/test_system_utter.txt
python3 bleu.py -pred_file ./output/perturb_kl.txt -label_file ./data/test_system_utter.txt
python3 bleu.py -pred_file ./output/perturb_all.txt -label_file ./data/test_system_utter.txt

# # calculate emotional valence
CUDA_VISIBLE_DEVICES=5 python3 emo_val_cal.py --user_file ./data/test_user_utter.txt --base_output ./output/DialoGPT.txt --val_output ./output/perturb_val.txt --rel_output ./output/perturb_rel.txt --kl_output ./output/perturb_kl.txt --all_output ./output/perturb_all.txt  --val_config ./models/EmoVal/bert_config/32220config.json --val_model ./models/EmoVal/bert_model/32220pytorch_model.bin

# # calculate information relevance
CUDA_VISIBLE_DEVICES=5 python3 inf_rel_cal.py --user_file ./data/test_user_utter.txt --base_output ./output/DialoGPT.txt --val_output ./output/perturb_val.txt --rel_output ./output/perturb_rel.txt --kl_output ./output/perturb_kl.txt --all_output ./output/perturb_all.txt  --rel_config ./models/InfRel/medium/config.json --rel_model ./models/InfRel/medium/ED-reverseV2-step-10000.pkl

: <<'END'
以baseline模型(使用EmpatheticDialogues微調過的DialoGPT)進行生成
程式: dialogGPT_generation.py
參數:
    out_dir 可指定輸出至output裡的其他資料夾(需事先建立好資料夾)
    finetune_generation_model 傳入經過finetune的DialoGPT模型.pkl檔(使用EmpatheticDialogues)
    for_test_run 為了測試程式能否順利跑完，設置跑完5句就結束

使用Plug-and-Play架構進行Empathy Perturbation，進行同理對話回應生成(Perturb_all版本)
程式: ED_pplm_sen_all.py
參數:
    out_dir 可指定輸出至output裡的其他資料夾(需事先建立好資料夾)
    length 最長輸出長度
    num_samples 擾動次數(代表重新生成幾次句子去做擾動)
    sample 原PPLM作者設置，不做更動，指從end-of-text做生成
    M 設定預設pretrained model(本實驗使用DialoGPT medium版本，有12+12層decode layer)
    verbosity 原PPLM作者設置，用來控制print要印出多少內容，設為quiet代表印出最少print
    finetune_generation_model 傳入經過finetune的DialoGPT模型.pkl檔(使用EmpatheticDialogues)
    emotional_valence_config 傳入Emotional Valence model的config
    emotional_valence_model 傳入Emotional Valence model的.bin檔案
    information_relevance_config 傳入Information Relevance model的config
    information_relevance_model 傳入Information Relevance model的.pkl檔案
    for_test_run 為了測試程式能否順利跑完，設置跑完5句就結束

使用Plug-and-Play架構進行Empathy Perturbation，進行同理對話回應生成(Perturb_rel版本)
程式: ED_pplm_sen_rel.py
參數:
    out_dir 可指定輸出至output裡的其他資料夾(需事先建立好資料夾)
    length 最長輸出長度
    num_samples 擾動次數(代表重新生成幾次句子去做擾動)
    sample 原PPLM作者設置，不做更動，指從end-of-text做生成
    M 設定預設pretrained model(本實驗使用DialoGPT medium版本，有12+12層decode layer)
    verbosity 原PPLM作者設置，用來控制print要印出多少內容，設為quiet代表印出最少print
    finetune_generation_model 傳入經過finetune的DialoGPT模型.pkl檔(使用EmpatheticDialogues)
    emotional_valence_config 傳入Emotional Valence model的config
    emotional_valence_model 傳入Emotional Valence model的.bin檔案
    information_relevance_config 傳入Information Relevance model的config
    information_relevance_model 傳入Information Relevance model的.pkl檔案
    for_test_run 為了測試程式能否順利跑完，設置跑完5句就結束

使用Plug-and-Play架構進行Empathy Perturbation，進行同理對話回應生成(Perturb_val版本)
程式: ED_pplm_sen_val.py
參數:
    out_dir 可指定輸出至output裡的其他資料夾(需事先建立好資料夾)
    length 最長輸出長度
    num_samples 擾動次數(代表重新生成幾次句子去做擾動)
    sample 原PPLM作者設置，不做更動，指從end-of-text做生成
    M 設定預設pretrained model(本實驗使用DialoGPT medium版本，有12+12層decode layer)
    verbosity 原PPLM作者設置，用來控制print要印出多少內容，設為quiet代表印出最少print
    finetune_generation_model 傳入經過finetune的DialoGPT模型.pkl檔(使用EmpatheticDialogues)
    emotional_valence_config 傳入Emotional Valence model的config
    emotional_valence_model 傳入Emotional Valence model的.bin檔案
    information_relevance_config 傳入Information Relevance model的config
    information_relevance_model 傳入Information Relevance model的.pkl檔案
    for_test_run 為了測試程式能否順利跑完，設置跑完5句就結束

使用Plug-and-Play架構進行Empathy Perturbation，進行同理對話回應生成(Perturb_kl版本)
程式: ED_pplm_sen_kl.py
參數:
    out_dir 可指定輸出至output裡的其他資料夾(需事先建立好資料夾)
    length 最長輸出長度
    num_samples 擾動次數(代表重新生成幾次句子去做擾動)
    sample 原PPLM作者設置，不做更動，指從end-of-text做生成
    M 設定預設pretrained model(本實驗使用DialoGPT medium版本，有12+12層decode layer)
    verbosity 原PPLM作者設置，用來控制print要印出多少內容，設為quiet代表印出最少print
    finetune_generation_model 傳入經過finetune的DialoGPT模型.pkl檔(使用EmpatheticDialogues)
    emotional_valence_config 傳入Emotional Valence model的config
    emotional_valence_model 傳入Emotional Valence model的.bin檔案
    information_relevance_config 傳入Information Relevance model的config
    information_relevance_model 傳入Information Relevance model的.pkl檔案
    for_test_run 為了測試程式能否順利跑完，設置跑完5句就結束

計算生成結果的Bleu score
程式: bleu.py
參數:
    pred_file 生成模型生成之回應句檔案
    label_file 語料原本的回應句檔案

計算生成結果的emotional valence
程式: emo_val_cal
參數:
    user_file 使用者語句
    base_output baseline生成結果
    val_output perturb_val生成結果
    rel_output perturb_rel生成結果
    kl_output perturb_kl生成結果
    all_output perturb_all生成結果
    val_config emotional valence偵測模型config
    val_model emotional valence偵測模型

計算生成結果的informaion relevance
程式: inf_rel_cal
參數:
    user_file 使用者語句
    base_output baseline生成結果
    val_output perturb_val生成結果
    rel_output perturb_rel生成結果
    kl_output perturb_kl生成結果
    all_output perturb_all生成結果
    rel_config informaion relevance偵測模型config
    rel_model informaion relevance偵測模型

由於PPLM原始架構為多層function結構，不易新增參數傳導至每一層，因此在不同設置實驗部分寫了基於perturb_all修改的各個版本，依照每個設置進行weight與loss的設置，各個版本也可利於同時分配到不同GPU上執行，節省從頭依序執行各版本時間。

另外emo_val_cal與inf_rel_cal，由於讀取模型需要一點時間，因此將所有生成結果寫成一次跑完計算，節省初始重複讀取模型時間，另外各個生成結果default皆為None，因此可藉由參數設定要傳哪個版本生成結果進去，可以依據現有結果傳入不用一次全部版本都傳。

***informaion relevance程式計算上需注意Transformer版本問題:
書瑀學長之前使用相似架構時有遇到程式上的輸出問題
原先程式為
loss_rel, _, _ = reverse_model(inputs_rel, labels=labels)
但有遇到跑不出loss分數的問題，只跑出loss標籤名。

後來學長發現可能是Transformer 3.x版後，模型的輸出不再是turple，而改為類似dict的物件，
裡面分別為loss, logits, past_key_value
所以要寫成
output = model() 
print(output.loss)
才能順利抓到loss
END