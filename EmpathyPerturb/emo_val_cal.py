#! /usr/bin/env python3
# coding=utf-8
'''
CUDA_VISIBLE_DEVICES=2 python3 emo_val_cal.py --user_file ./data/test_user_utter.txt --base_output ./output/DialoGPT.txt --val_output ./output/perturb_val.txt --rel_output ./output/perturb_rel.txt --kl_output ./output/perturb_kl.txt --all_output ./output/perturb_all.txt  --val_config ./models/EmoVal/bert_config/32220config.json --val_model ./models/EmoVal/bert_model/32220pytorch_model.bin
'''
import argparse
import sys, os
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer, GPT2Config
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertForSequenceClassification, BertModel
from torch import nn


class SequenceClassification(BertForSequenceClassification):
    def __init__(self, config, num_labels):
        super(SequenceClassification, self).__init__(config, num_labels)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                logits = torch.sigmoid(logits)
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
                return pooled_output, loss
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return pooled_output, loss
        else:
            return pooled_output, logits

def preprocess_detect(inputs_id, device):
    segment_ids = torch.tensor([[0 for word_id in input_id] for input_id in inputs_id], device=device, dtype=torch.long)
    input_mask = torch.tensor([[1 if word_id==1 else 0 for word_id in input_id] for input_id in inputs_id], device=device, dtype=torch.long)
    return segment_ids, input_mask

def preprocess_valence(tokens_a, tokens_b):
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b != None:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
    else:
        tokens += ["[SEP]"]
        segment_ids += [1] * 1

    input_mask = [1] * len(tokens)

    return segment_ids, input_mask, tokens

parser = argparse.ArgumentParser()

parser.add_argument('--user_file', type=str, required=True)
parser.add_argument('--base_output', type=str, default=None)
parser.add_argument('--val_output', type=str, default=None)
parser.add_argument('--rel_output', type=str, default=None)
parser.add_argument('--kl_output', type=str, default=None)
parser.add_argument('--all_output', type=str, default=None)
parser.add_argument('--val_config', type=str, required=True)
parser.add_argument('--val_model', type=str, required=True)

opt = parser.parse_args()
end_t = " <|endoftext|>"

# set Random seed
torch.manual_seed(seed=23)
np.random.seed(seed=23)

# set the device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

config_detect = BertConfig(opt.val_config)
detect_model = SequenceClassification(config_detect, num_labels=1).to(device)
detect_model.load_state_dict(torch.load(opt.val_model))
detect_model.eval()


bert_tokenizer = BertTokenizer(vocab_file='./vocab/bert-base-uncased-vocab.txt')

path_user = opt.user_file

# # # === base_output calulate part =====================================
if opt.base_output != None:
    path_response = opt.base_output

    response_file = path_response.split('/')[-1]

    path_output = './emotional_valence/EmoVal_' + response_file
    print("response_file: ", path_output)

    f_user = open(path_user, 'r', encoding='utf-8')
    f_system = open(path_response, 'r', encoding='utf-8')
    f_output = open(path_output, 'w', encoding='utf-8')

    content_user = []
    content_system = []

    line_count = 0
    for line in f_user:
        line_count += 1
        if line.strip():
            line=line.strip()
            content_user.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")

    line_count = 0
    for line in f_system:
        line_count += 1
        if line.strip():
            line=line.strip()
            content_system.append(line + end_t)
        else:
            print("===Warning: " + str(line_count) + " line===")
            content_system.append(end_t)

    print(len(content_user))
    print(len(content_system))
    if len(content_user) < len(content_system):   
        length = len(content_user)
    else:
        length = len(content_system)

    maxi = 0.0
    mini = 100.0
    emo_val_count = 0.0
    num_count = 0
    for i in range(length):
        mse_loss = torch.nn.MSELoss()
        
        token_a = bert_tokenizer.tokenize(content_user[i])
        token_b = bert_tokenizer.tokenize(content_system[i])

        bert_encode = ["[CLS]"] + token_a + ["[SEP]"]
        sentence_ids = bert_tokenizer.convert_tokens_to_ids(bert_encode)
        input_tensor = torch.tensor(sentence_ids, device=device, dtype=torch.long)
        input_tensor = input_tensor.unsqueeze(0)

        detect_segment_id, detect_mask = preprocess_detect(input_tensor, device)
        detect_pooled_output, detect_logits = detect_model(input_tensor, detect_segment_id, detect_mask)
        user_valence = torch.sigmoid(detect_logits)

        bert_encode = ["[CLS]"] + token_b + ["[SEP]"]
        sentence_ids = bert_tokenizer.convert_tokens_to_ids(bert_encode)
        input_tensor = torch.tensor(sentence_ids, device=device, dtype=torch.long)
        input_tensor = input_tensor.unsqueeze(0)

        detect_segment_id, detect_mask = preprocess_detect(input_tensor, device)
        detect_pooled_output, detect_logits = detect_model(input_tensor, detect_segment_id, detect_mask)
        system_valence = torch.sigmoid(detect_logits)

        val_loss = mse_loss(system_valence, user_valence)
        val_loss = torch.sqrt(val_loss)

        emo_val_loss = float(val_loss)

        emo_val_count += emo_val_loss

        if emo_val_loss >= maxi:
            maxi = emo_val_loss
        if emo_val_loss <= mini:
            mini = emo_val_loss

        # f_output.write(str(emo_val_loss) + '\n')

    avg_rel = emo_val_count / len(content_system)
    print("=====avg emo_val_loss: ", avg_rel)
    print("maxi emo_val_loss: ", maxi)
    print("mini emo_val_loss: ", mini)

    f_output.write('file\t{}\tavg emo_val_loss: {}\tmaxi emo_val_loss: {}\tmini emo_val_loss: {}\n'.format(path_response, avg_rel, maxi, mini))

    f_user.close()
    f_system.close()
    f_output.close()
else:
    print("None base output!")

# # # === val_output calulate part =====================================
if opt.val_output != None:
    path_response = opt.val_output

    response_file = path_response.split('/')[-1]

    path_output = './emotional_valence/EmoVal_' + response_file
    print("response_file: ", path_output)

    f_user = open(path_user, 'r', encoding='utf-8')
    f_system = open(path_response, 'r', encoding='utf-8')
    f_output = open(path_output, 'w', encoding='utf-8')

    content_user = []
    content_system = []

    line_count = 0
    for line in f_user:
        line_count += 1
        if line.strip():
            line=line.strip()
            content_user.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")

    line_count = 0
    for line in f_system:
        line_count += 1
        if line.strip():
            line=line.strip()
            content_system.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")
            content_system.append(end_t)

    print(len(content_user))
    print(len(content_system))
    if len(content_user) < len(content_system):   
        length = len(content_user)
    else:
        length = len(content_system)

    maxi = 0.0
    mini = 100.0
    emo_val_count = 0.0
    num_count = 0
    for i in range(length):
        mse_loss = torch.nn.MSELoss()
        
        token_a = bert_tokenizer.tokenize(content_user[i])
        token_b = bert_tokenizer.tokenize(content_system[i])

        bert_encode = ["[CLS]"] + token_a + ["[SEP]"]
        sentence_ids = bert_tokenizer.convert_tokens_to_ids(bert_encode)
        input_tensor = torch.tensor(sentence_ids, device=device, dtype=torch.long)
        input_tensor = input_tensor.unsqueeze(0)

        detect_segment_id, detect_mask = preprocess_detect(input_tensor, device)
        detect_pooled_output, detect_logits = detect_model(input_tensor, detect_segment_id, detect_mask)
        user_valence = torch.sigmoid(detect_logits)

        bert_encode = ["[CLS]"] + token_b + ["[SEP]"]
        sentence_ids = bert_tokenizer.convert_tokens_to_ids(bert_encode)
        input_tensor = torch.tensor(sentence_ids, device=device, dtype=torch.long)
        input_tensor = input_tensor.unsqueeze(0)

        detect_segment_id, detect_mask = preprocess_detect(input_tensor, device)
        detect_pooled_output, detect_logits = detect_model(input_tensor, detect_segment_id, detect_mask)
        system_valence = torch.sigmoid(detect_logits)

        val_loss = mse_loss(system_valence, user_valence)
        val_loss = torch.sqrt(val_loss)

        emo_val_loss = float(val_loss)

        emo_val_count += emo_val_loss

        if emo_val_loss >= maxi:
            maxi = emo_val_loss
        if emo_val_loss <= mini:
            mini = emo_val_loss

        # f_output.write(str(emo_val_loss) + '\n')

    avg_rel = emo_val_count / len(content_system)
    print("=====avg emo_val_loss: ", avg_rel)
    print("maxi emo_val_loss: ", maxi)
    print("mini emo_val_loss: ", mini)

    f_output.write('file\t{}\tavg emo_val_loss: {}\tmaxi emo_val_loss: {}\tmini emo_val_loss: {}\n'.format(path_response, avg_rel, maxi, mini))

    f_user.close()
    f_system.close()
    f_output.close()
else:
    print("None val output!")

# # # === rel_output calulate part =====================================
if opt.rel_output != None:
    path_response = opt.rel_output

    response_file = path_response.split('/')[-1]

    path_output = './emotional_valence/EmoVal_' + response_file
    print("response_file: ", path_output)

    f_user = open(path_user, 'r', encoding='utf-8')
    f_system = open(path_response, 'r', encoding='utf-8')
    f_output = open(path_output, 'w', encoding='utf-8')

    content_user = []
    content_system = []

    line_count = 0
    for line in f_user:
        line_count += 1
        if line.strip():
            line=line.strip()
            content_user.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")

    line_count = 0
    for line in f_system:
        line_count += 1
        if line.strip():
            line=line.strip()
            content_system.append(line + end_t)
        else:
            print("===Warning: " + str(line_count) + " line===")
            content_system.append(end_t)

    print(len(content_user))
    print(len(content_system))
    if len(content_user) < len(content_system):   
        length = len(content_user)
    else:
        length = len(content_system)

    maxi = 0.0
    mini = 100.0
    emo_val_count = 0.0
    num_count = 0
    for i in range(length):
        mse_loss = torch.nn.MSELoss()
        
        token_a = bert_tokenizer.tokenize(content_user[i])
        token_b = bert_tokenizer.tokenize(content_system[i])

        bert_encode = ["[CLS]"] + token_a + ["[SEP]"]
        sentence_ids = bert_tokenizer.convert_tokens_to_ids(bert_encode)
        input_tensor = torch.tensor(sentence_ids, device=device, dtype=torch.long)
        input_tensor = input_tensor.unsqueeze(0)

        detect_segment_id, detect_mask = preprocess_detect(input_tensor, device)
        detect_pooled_output, detect_logits = detect_model(input_tensor, detect_segment_id, detect_mask)
        user_valence = torch.sigmoid(detect_logits)

        bert_encode = ["[CLS]"] + token_b + ["[SEP]"]
        sentence_ids = bert_tokenizer.convert_tokens_to_ids(bert_encode)
        input_tensor = torch.tensor(sentence_ids, device=device, dtype=torch.long)
        input_tensor = input_tensor.unsqueeze(0)

        detect_segment_id, detect_mask = preprocess_detect(input_tensor, device)
        detect_pooled_output, detect_logits = detect_model(input_tensor, detect_segment_id, detect_mask)
        system_valence = torch.sigmoid(detect_logits)

        val_loss = mse_loss(system_valence, user_valence)
        val_loss = torch.sqrt(val_loss)

        emo_val_loss = float(val_loss)

        emo_val_count += emo_val_loss

        if emo_val_loss >= maxi:
            maxi = emo_val_loss
        if emo_val_loss <= mini:
            mini = emo_val_loss

        # f_output.write(str(emo_val_loss) + '\n')

    avg_rel = emo_val_count / len(content_system)
    print("=====avg emo_val_loss: ", avg_rel)
    print("maxi emo_val_loss: ", maxi)
    print("mini emo_val_loss: ", mini)

    f_output.write('file\t{}\tavg emo_val_loss: {}\tmaxi emo_val_loss: {}\tmini emo_val_loss: {}\n'.format(path_response, avg_rel, maxi, mini))

    f_user.close()
    f_system.close()
    f_output.close()
else:
    print("None rel output!")

# # # === kl_output calulate part =====================================
if opt.kl_output != None:
    path_response = opt.kl_output

    response_file = path_response.split('/')[-1]

    path_output = './emotional_valence/EmoVal_' + response_file
    print("response_file: ", path_output)

    f_user = open(path_user, 'r', encoding='utf-8')
    f_system = open(path_response, 'r', encoding='utf-8')
    f_output = open(path_output, 'w', encoding='utf-8')

    content_user = []
    content_system = []

    line_count = 0
    for line in f_user:
        line_count += 1
        if line.strip():
            line=line.strip()
            content_user.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")

    line_count = 0
    for line in f_system:
        line_count += 1
        if line.strip():
            line=line.strip()
            content_system.append(line + end_t)
        else:
            print("===Warning: " + str(line_count) + " line===")
            content_system.append(end_t)

    print(len(content_user))
    print(len(content_system))
    if len(content_user) < len(content_system):   
        length = len(content_user)
    else:
        length = len(content_system)

    maxi = 0.0
    mini = 100.0
    emo_val_count = 0.0
    num_count = 0
    for i in range(length):
        mse_loss = torch.nn.MSELoss()
        
        token_a = bert_tokenizer.tokenize(content_user[i])
        token_b = bert_tokenizer.tokenize(content_system[i])

        bert_encode = ["[CLS]"] + token_a + ["[SEP]"]
        sentence_ids = bert_tokenizer.convert_tokens_to_ids(bert_encode)
        input_tensor = torch.tensor(sentence_ids, device=device, dtype=torch.long)
        input_tensor = input_tensor.unsqueeze(0)

        detect_segment_id, detect_mask = preprocess_detect(input_tensor, device)
        detect_pooled_output, detect_logits = detect_model(input_tensor, detect_segment_id, detect_mask)
        user_valence = torch.sigmoid(detect_logits)

        bert_encode = ["[CLS]"] + token_b + ["[SEP]"]
        sentence_ids = bert_tokenizer.convert_tokens_to_ids(bert_encode)
        input_tensor = torch.tensor(sentence_ids, device=device, dtype=torch.long)
        input_tensor = input_tensor.unsqueeze(0)

        detect_segment_id, detect_mask = preprocess_detect(input_tensor, device)
        detect_pooled_output, detect_logits = detect_model(input_tensor, detect_segment_id, detect_mask)
        system_valence = torch.sigmoid(detect_logits)

        val_loss = mse_loss(system_valence, user_valence)
        val_loss = torch.sqrt(val_loss)

        emo_val_loss = float(val_loss)

        emo_val_count += emo_val_loss

        if emo_val_loss >= maxi:
            maxi = emo_val_loss
        if emo_val_loss <= mini:
            mini = emo_val_loss

        # f_output.write(str(emo_val_loss) + '\n')

    avg_rel = emo_val_count / len(content_system)
    print("=====avg emo_val_loss: ", avg_rel)
    print("maxi emo_val_loss: ", maxi)
    print("mini emo_val_loss: ", mini)

    f_output.write('file\t{}\tavg emo_val_loss: {}\tmaxi emo_val_loss: {}\tmini emo_val_loss: {}\n'.format(path_response, avg_rel, maxi, mini))

    f_user.close()
    f_system.close()
    f_output.close()
else:
    print("None kl output!")

# # # === all_output calulate part =====================================
if opt.all_output != None:
    path_response = opt.all_output

    response_file = path_response.split('/')[-1]

    path_output = './emotional_valence/EmoVal_' + response_file
    print("response_file: ", path_output)

    f_user = open(path_user, 'r', encoding='utf-8')
    f_system = open(path_response, 'r', encoding='utf-8')
    f_output = open(path_output, 'w', encoding='utf-8')

    content_user = []
    content_system = []

    line_count = 0
    for line in f_user:
        line_count += 1
        if line.strip():
            line=line.strip()
            content_user.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")

    line_count = 0
    for line in f_system:
        line_count += 1
        if line.strip():
            line=line.strip()
            content_system.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")
            content_system.append(end_t)

    print(len(content_user))
    print(len(content_system))
    if len(content_user) < len(content_system):   
        length = len(content_user)
    else:
        length = len(content_system)

    maxi = 0.0
    mini = 100.0
    emo_val_count = 0.0
    num_count = 0
    for i in range(length):
        mse_loss = torch.nn.MSELoss()
        
        token_a = bert_tokenizer.tokenize(content_user[i])
        token_b = bert_tokenizer.tokenize(content_system[i])

        bert_encode = ["[CLS]"] + token_a + ["[SEP]"]
        sentence_ids = bert_tokenizer.convert_tokens_to_ids(bert_encode)
        input_tensor = torch.tensor(sentence_ids, device=device, dtype=torch.long)
        input_tensor = input_tensor.unsqueeze(0)

        detect_segment_id, detect_mask = preprocess_detect(input_tensor, device)
        detect_pooled_output, detect_logits = detect_model(input_tensor, detect_segment_id, detect_mask)
        user_valence = torch.sigmoid(detect_logits)

        bert_encode = ["[CLS]"] + token_b + ["[SEP]"]
        sentence_ids = bert_tokenizer.convert_tokens_to_ids(bert_encode)
        input_tensor = torch.tensor(sentence_ids, device=device, dtype=torch.long)
        input_tensor = input_tensor.unsqueeze(0)

        detect_segment_id, detect_mask = preprocess_detect(input_tensor, device)
        detect_pooled_output, detect_logits = detect_model(input_tensor, detect_segment_id, detect_mask)
        system_valence = torch.sigmoid(detect_logits)

        val_loss = mse_loss(system_valence, user_valence)
        val_loss = torch.sqrt(val_loss)

        emo_val_loss = float(val_loss)

        emo_val_count += emo_val_loss

        if emo_val_loss >= maxi:
            maxi = emo_val_loss
        if emo_val_loss <= mini:
            mini = emo_val_loss

        # f_output.write(str(emo_val_loss) + '\n')

    avg_rel = emo_val_count / len(content_system)
    print("=====avg emo_val_loss: ", avg_rel)
    print("maxi emo_val_loss: ", maxi)
    print("mini emo_val_loss: ", mini)

    f_output.write('file\t{}\tavg emo_val_loss: {}\tmaxi emo_val_loss: {}\tmini emo_val_loss: {}\n'.format(path_response, avg_rel, maxi, mini))

    f_user.close()
    f_system.close()
    f_output.close()
else:
    print("None all output!")