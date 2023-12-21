#! /usr/bin/env python3
# coding=utf-8
'''
CUDA_VISIBLE_DEVICES=3 python3 inf_rel_cal.py --user_file ./data/test_user_utter.txt --base_output ./output/DialoGPT.txt --val_output ./output/perturb_val.txt --rel_output ./output/perturb_rel.txt --kl_output ./output/perturb_kl.txt --all_output ./output/perturb_all.txt  --rel_config ./models/InfRel/medium/config.json --rel_model ./models/InfRel/medium/ED-reverseV2-step-10000.pkl
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

parser = argparse.ArgumentParser()

parser.add_argument('--user_file', type=str, required=True)
parser.add_argument('--base_output', type=str, default=None)
parser.add_argument('--val_output', type=str, default=None)
parser.add_argument('--rel_output', type=str, default=None)
parser.add_argument('--kl_output', type=str, default=None)
parser.add_argument('--all_output', type=str, default=None)
parser.add_argument('--rel_config', type=str, required=True)
parser.add_argument('--rel_model', type=str,required=True)

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

cfg = GPT2Config.from_json_file(opt.rel_config)

weights = torch.load(opt.rel_model)

# fix misused key value
weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
weights.pop("lm_head.decoder.weight", None)

reverse_model=GPT2LMHeadModel(cfg)
reverse_model.load_state_dict(weights, strict=False)
reverse_model.to(device)
reverse_model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")

path_user = opt.user_file

# # # === base_output calulate part =====================================
if opt.base_output != None:
    path_response = opt.base_output

    response_file = path_response.split('/')[-1]

    path_output = './information_relevance/InfRel_' + response_file
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
            line=line.replace("<|endoftext|>", "")
            line=line.strip()
            content_user.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")

    line_count = 0
    for line in f_system:
        line_count += 1
        if line.strip():
            line=line.replace("<|endoftext|>", "")
            line=line.strip()
            content_system.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")
            content_system.append(end_t)

    # print(len(content_user))
    # print(len(content_system))
    if len(content_user) < len(content_system):   
        length = len(content_user)
    else:
        length = len(content_system)

    maxi = 0.0
    mini = 100.0
    rel_count = 0.0
    num_count = 0
    for i in range(length):
        last_response = tokenizer.encode(
                content_system[i],
                add_special_tokens=False
            )
        last_response_t = torch.tensor(last_response, device=device, dtype=torch.long)
        while len(last_response_t.shape) < 2:
            last_response_t = last_response_t.unsqueeze(0)

        context = tokenizer.encode(
                content_user[i],
                add_special_tokens=False
            )
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)

        inputs_rel = torch.cat((last_response_t, context_t), dim=1)
        mask_rel = torch.full_like(last_response_t, -100, dtype=torch.long)
        labels = torch.cat((mask_rel, context_t), dim=1)

        loss_rel, _, _ = reverse_model(inputs_rel, labels=labels)
        rel_loss = float(loss_rel)

        rel_count += rel_loss

        if rel_loss >= maxi:
            maxi = rel_loss
        if rel_loss <= mini:
            mini = rel_loss

        # f_output.write(str(rel_loss) + '\n')

    avg_rel = rel_count / len(content_system)
    print("=====avg rel_loss: ", avg_rel)
    print("maxi rel_loss: ", maxi)
    print("mini rel_loss: ", mini)

    f_output.write('file\t{}\tavg rel_loss: {}\tmaxi rel_loss: {}\tmini rel_loss: {}\n'.format(path_response, avg_rel, maxi, mini))

    f_user.close()
    f_system.close()
    f_output.close()
else:
    print("None base output!")

# # # === val_output calulate part =====================================
if opt.val_output != None:
    path_response = opt.val_output

    response_file = path_response.split('/')[-1]

    path_output = './information_relevance/InfRel_' + response_file
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
            line=line.replace("<|endoftext|>", "")
            line=line.strip()
            content_user.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")

    line_count = 0
    for line in f_system:
        line_count += 1
        if line.strip():
            line=line.replace("<|endoftext|>", "")
            line=line.strip()
            content_system.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")
            content_system.append(end_t)

    # print(len(content_user))
    # print(len(content_system))
    if len(content_user) < len(content_system):   
        length = len(content_user)
    else:
        length = len(content_system)

    maxi = 0.0
    mini = 100.0
    rel_count = 0.0
    num_count = 0
    for i in range(length):
        last_response = tokenizer.encode(
                content_system[i],
                add_special_tokens=False
            )
        last_response_t = torch.tensor(last_response, device=device, dtype=torch.long)
        while len(last_response_t.shape) < 2:
            last_response_t = last_response_t.unsqueeze(0)

        context = tokenizer.encode(
                content_user[i],
                add_special_tokens=False
            )
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)

        inputs_rel = torch.cat((last_response_t, context_t), dim=1)
        mask_rel = torch.full_like(last_response_t, -100, dtype=torch.long)
        labels = torch.cat((mask_rel, context_t), dim=1)

        loss_rel, _, _ = reverse_model(inputs_rel, labels=labels)
        rel_loss = float(loss_rel)

        rel_count += rel_loss

        if rel_loss >= maxi:
            maxi = rel_loss
        if rel_loss <= mini:
            mini = rel_loss

        # f_output.write(str(rel_loss) + '\n')

    avg_rel = rel_count / len(content_system)
    print("=====avg rel_loss: ", avg_rel)
    print("maxi rel_loss: ", maxi)
    print("mini rel_loss: ", mini)

    f_output.write('file\t{}\tavg rel_loss: {}\tmaxi rel_loss: {}\tmini rel_loss: {}\n'.format(path_response, avg_rel, maxi, mini))

    f_user.close()
    f_system.close()
    f_output.close()
else:
    print("None val output!")

# # # === rel_output calulate part =====================================
if opt.rel_output != None:
    path_response = opt.rel_output

    response_file = path_response.split('/')[-1]

    path_output = './information_relevance/InfRel_' + response_file
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
            line=line.replace("<|endoftext|>", "")
            line=line.strip()
            content_user.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")

    line_count = 0
    for line in f_system:
        line_count += 1
        if line.strip():
            line=line.replace("<|endoftext|>", "")
            line=line.strip()
            content_system.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")
            content_system.append(end_t)

    # print(len(content_user))
    # print(len(content_system))
    if len(content_user) < len(content_system):   
        length = len(content_user)
    else:
        length = len(content_system)

    maxi = 0.0
    mini = 100.0
    rel_count = 0.0
    num_count = 0
    for i in range(length):
        last_response = tokenizer.encode(
                content_system[i],
                add_special_tokens=False
            )
        last_response_t = torch.tensor(last_response, device=device, dtype=torch.long)
        while len(last_response_t.shape) < 2:
            last_response_t = last_response_t.unsqueeze(0)

        context = tokenizer.encode(
                content_user[i] + "<|endoftext|>",
                add_special_tokens=False
            )
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)

        inputs_rel = torch.cat((last_response_t, context_t), dim=1)
        mask_rel = torch.full_like(last_response_t, -100, dtype=torch.long)
        labels = torch.cat((mask_rel, context_t), dim=1)

        loss_rel, _, _ = reverse_model(inputs_rel, labels=labels)
        rel_loss = float(loss_rel)

        rel_count += rel_loss

        if rel_loss >= maxi:
            maxi = rel_loss
        if rel_loss <= mini:
            mini = rel_loss

        # f_output.write(str(rel_loss) + '\n')

    avg_rel = rel_count / len(content_system)
    print("=====avg rel_loss: ", avg_rel)
    print("maxi rel_loss: ", maxi)
    print("mini rel_loss: ", mini)

    f_output.write('file\t{}\tavg rel_loss: {}\tmaxi rel_loss: {}\tmini rel_loss: {}\n'.format(path_response, avg_rel, maxi, mini))

    f_user.close()
    f_system.close()
    f_output.close()
else:
    print("None rel output!")

# # # === kl_output calulate part =====================================
if opt.kl_output != None:
    path_response = opt.kl_output

    response_file = path_response.split('/')[-1]

    path_output = './information_relevance/InfRel_' + response_file
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
            line=line.replace("<|endoftext|>", "")
            line=line.strip()
            content_user.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")

    line_count = 0
    for line in f_system:
        line_count += 1
        if line.strip():
            line=line.replace("<|endoftext|>", "")
            line=line.strip()
            content_system.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")
            content_system.append(end_t)

    # print(len(content_user))
    # print(len(content_system))
    if len(content_user) < len(content_system):   
        length = len(content_user)
    else:
        length = len(content_system)

    maxi = 0.0
    mini = 100.0
    rel_count = 0.0
    num_count = 0
    for i in range(length):
        last_response = tokenizer.encode(
                content_system[i],
                add_special_tokens=False
            )
        last_response_t = torch.tensor(last_response, device=device, dtype=torch.long)
        while len(last_response_t.shape) < 2:
            last_response_t = last_response_t.unsqueeze(0)

        context = tokenizer.encode(
                content_user[i],
                add_special_tokens=False
            )
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)

        inputs_rel = torch.cat((last_response_t, context_t), dim=1)
        mask_rel = torch.full_like(last_response_t, -100, dtype=torch.long)
        labels = torch.cat((mask_rel, context_t), dim=1)

        loss_rel, _, _ = reverse_model(inputs_rel, labels=labels)
        rel_loss = float(loss_rel)

        rel_count += rel_loss

        if rel_loss >= maxi:
            maxi = rel_loss
        if rel_loss <= mini:
            mini = rel_loss

        # f_output.write(str(rel_loss) + '\n')

    avg_rel = rel_count / len(content_system)
    print("=====avg rel_loss: ", avg_rel)
    print("maxi rel_loss: ", maxi)
    print("mini rel_loss: ", mini)

    f_output.write('file\t{}\tavg rel_loss: {}\tmaxi rel_loss: {}\tmini rel_loss: {}\n'.format(path_response, avg_rel, maxi, mini))

    f_user.close()
    f_system.close()
    f_output.close()
else:
    print("None kl output!")

# # # === all_output calulate part =====================================
if opt.all_output != None:
    path_response = opt.all_output

    response_file = path_response.split('/')[-1]

    path_output = './information_relevance/InfRel_' + response_file
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
            line=line.replace("<|endoftext|>", "")
            line=line.strip()
            content_user.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")

    line_count = 0
    for line in f_system:
        line_count += 1
        if line.strip():
            line=line.replace("<|endoftext|>", "")
            line=line.strip()
            content_system.append(line)
        else:
            print("===Warning: " + str(line_count) + " line===")
            content_system.append(end_t)

    # print(len(content_user))
    # print(len(content_system))
    if len(content_user) < len(content_system):   
        length = len(content_user)
    else:
        length = len(content_system)

    maxi = 0.0
    mini = 100.0
    rel_count = 0.0
    num_count = 0
    for i in range(length):
        last_response = tokenizer.encode(
                content_system[i],
                add_special_tokens=False
            )
        last_response_t = torch.tensor(last_response, device=device, dtype=torch.long)
        while len(last_response_t.shape) < 2:
            last_response_t = last_response_t.unsqueeze(0)

        context = tokenizer.encode(
                content_user[i] + "<|endoftext|>",
                add_special_tokens=False
            )
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)

        inputs_rel = torch.cat((last_response_t, context_t), dim=1)
        mask_rel = torch.full_like(last_response_t, -100, dtype=torch.long)
        labels = torch.cat((mask_rel, context_t), dim=1)

        loss_rel, _, _ = reverse_model(inputs_rel, labels=labels)
        rel_loss = float(loss_rel)

        rel_count += rel_loss

        if rel_loss >= maxi:
            maxi = rel_loss
        if rel_loss <= mini:
            mini = rel_loss

        # f_output.write(str(rel_loss) + '\n')

    avg_rel = rel_count / len(content_system)
    print("=====avg rel_loss: ", avg_rel)
    print("maxi rel_loss: ", maxi)
    print("mini rel_loss: ", mini)

    f_output.write('file\t{}\tavg rel_loss: {}\tmaxi rel_loss: {}\tmini rel_loss: {}\n'.format(path_response, avg_rel, maxi, mini))

    f_user.close()
    f_system.close()
    f_output.close()
else:
    print("None all output!")