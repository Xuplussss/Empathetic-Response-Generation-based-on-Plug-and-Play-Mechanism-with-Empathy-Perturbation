#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CUDA_VISIBLE_DEVICES=1 python3 ED_pplm_sen_all.py --out_dir fortest --length 80 --num_samples 1 --sample -M microsoft/DialoGPT-medium --verbosity quiet --finetune_generation_model ./models/GenModel/ED-pretrain-step-10000.pkl --emotional_valence_config ./models/EmoVal/bert_config/32220config.json --emotional_valence_model ./models/EmoVal/bert_model/32220pytorch_model.bin --information_relevance_config ./models/InfRel/medium/config.json --information_relevance_model ./models/InfRel/medium/ED-reverseV2-step-10000.pkl
"""
import time
import sys, os
import argparse
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

# # for DialogGPT
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# # for valence and information model
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertForSequenceClassification, BertModel
from torch import nn

### about loss record
file_loss = None
rel_loss_record = 0.0
val_loss_record = 0.0
kl_loss_record = 0.0
pplm_loss_record = 0.0

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}


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


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_hidden(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR,
        output_so_far=None,
        tokenizer=None,
        bert_tokenizer=None,
        last_response=None,
        reverse_model=None,
        detect_model=None,
        context=None
):
    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))        
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        global rel_loss_record
        global val_loss_record
        global kl_loss_record
        global pplm_loss_record

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            max_rel = 13.0
            min_rel = 0.0

            end_token = torch.tensor([[50256]], device=device, dtype=torch.long)
            respon = last_response
            if next_token.tolist()[0][0] <= 50256:
                respon = (
                    next_token if last_response is None
                    else last_response
                )
            
            inputs_mmi = torch.cat((respon, context), dim=1)
            mask_mmi = torch.full_like(respon, -100, dtype=torch.long)
            labels = torch.cat((mask_mmi, context), dim=1)

            loss_mmi, _, _ = reverse_model(inputs_mmi, labels=labels)
            loss_rel = loss_mmi / max_rel

            if loss_rel.item() > 1.0:
                loss_rel = torch.div(loss_rel, loss_rel)
                print("loss rel > 1.0: ", loss_rel)
            elif loss_rel.item() < 0.0:
                loss_rel = torch.sub(loss_rel, loss_rel)
                print("loss rel < 0.0: ", loss_rel)

            # # # weight rel
            loss_rel = torch.mul(loss_rel, 0.4, out=None)
            
            if verbosity_level >= VERY_VERBOSE:
                print(" loss_mmi:", loss_mmi.data.cpu().numpy())

            loss += loss_rel
            rel_loss_record = loss_rel          
            loss_list.append(loss_rel)
            
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_rel_loss:", loss.data.cpu().numpy())

        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            
            mse_loss = torch.nn.MSELoss()

            token_a = tokenizer.decode(context.tolist()[0][:-1])
            token_a = bert_tokenizer.tokenize(token_a)
            
            token_b = last_response
            token_b = tokenizer.decode(token_b.tolist()[0])
            token_b = bert_tokenizer.tokenize(token_b)
            
            ### ===detection model begin===
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

            # # # weight val
            val_loss = torch.mul(val_loss, 0.4, out=None)
            ### ===detection model end===

            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_val_loss:", val_loss.data.cpu().numpy())

            val_loss_record = val_loss
            loss += val_loss.item()
            loss_list.append(val_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

            kl_loss_record = kl_loss
            pplm_loss_record = loss
            if file_loss != None:
                file_loss.write(str(float(rel_loss_record))+'\t'+str(float(val_loss_record))+'\t'+str(float(kl_loss_record))+'\t'+str(float(pplm_loss_record))+' \n')

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # if loss_type == PPLM_BOW:
        #     loss = val_loss + kl_loss
        # elif loss_type == PPLM_DISCRIM:
        #     loss = discrim_loss + kl_loss
        # elif loss_type == PPLM_BOW_DISCRIM:
        #     loss = val_loss + discrim_loss + kl_loss
        # else:
        #     print("loss error occured: ", loss)
        # loss_per_iter.append(loss.data.cpu().numpy())
        # print("loss: ", loss)

        # compute gradients
        loss.backward()
        
        if grad_norms is None or loss_type == PPLM_BOW_DISCRIM:
            grad_norms = [
                    (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                    for index, p_ in enumerate(curr_perturbation)
            ]
        elif grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        elif grad_norms is not None and loss_type == PPLM_DISCRIM:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + 2 * SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]
            
        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]
        
        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def full_text_generation_sentence_level(
        model,
        tokenizer,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        # class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        bert_tokenizer=None,
        reverse_model=None,
        detect_model=None,
        **kwargs
):
    # if bag_of_words and discrim:
    #     loss_type = PPLM_BOW_DISCRIM
    #     if verbosity_level >= REGULAR:
    #         print("Both PPLM-BoW and PPLM-Discrim are on. "
    #               "This is not optimized.")

    # elif bag_of_words:
    #     loss_type = PPLM_BOW
    #     if verbosity_level >= REGULAR:
    #         print("Using PPLM-BoW")

    # elif discrim is not None:
    #     loss_type = PPLM_DISCRIM
    #     if verbosity_level >= REGULAR:
    #         print("Using PPLM-Discrim")

    # else:
    #     raise Exception("Specify either a bag of words or a discriminator")

    # the flag for EmoVal and InfRel perturbation use original structure
    loss_type = PPLM_BOW_DISCRIM
    if verbosity_level >= REGULAR:
        print("PPLM for Empathy is on. "
                "This is not optimized.")

    unpert_gen_tok_text = None
    original_response = None
    last_response = None

    unpert_gen_tok_text, _, _, original_response = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        verbosity_level=verbosity_level
    )

    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    # last_response is the complete sentence for sentence level perturbation
    last_response = original_response
    
    # we first use last_response to perturb current hidden, and use perturbed hidden to generate a complete sentence
    for i in range(num_samples):
        # perturb current hidden to get pert_sen_past for sentence level perturbation and generation(different from original author version)
        pert_sen_past = perturb_sentence(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level,
            bert_tokenizer=bert_tokenizer,
            last_response=last_response,
            reverse_model=reverse_model,
            detect_model=detect_model
        )

        # use pert_sen_past to generate a complete sentence
        # here para_perturb = false, which means this function will use para_past = pert_sen_past  to generate a complete sentence without perturbation in word level
        # if para_perturb = true, this function will perturb in word level(original author design)
        pert_gen_tok_text, discrim_loss, loss_in_time, pert_response = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=False,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level,
            bert_tokenizer=bert_tokenizer,
            last_response=original_response,
            reverse_model=reverse_model,
            detect_model=detect_model,
            past=pert_sen_past
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        losses_in_time.append(loss_in_time)

        last_response = pert_response

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time, original_response, pert_response


def perturb_sentence(
        model,
        tokenizer,
        context=None,
        past=None,
        device="cuda",
        perturb=True,
        loss_type=0,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        bert_tokenizer=None,
        last_response=None,
        reverse_model=None,
        detect_model=None
):
    output_so_far = None
    system_response =None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    # Get past/probs for current output, except for last word
    # Note that GPT takes 2 inputs: past + current_token

    # run model forward to obtain unperturbed
    if past is None and output_so_far is not None:
        last = output_so_far[:, -1:]
        if output_so_far.shape[1] > 1:
            _, past, _ = model(output_so_far[:, :-1])

    unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
    unpert_last_hidden = unpert_all_hidden[-1]
     
    current_stepsize = stepsize

    # modify the past if necessary
    if not perturb or num_iterations == 0:
        pert_past = past

    else:
        accumulated_hidden = unpert_last_hidden[:, :-1, :]
        accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

        if past is not None:
            pert_past, _, grad_norms, loss_this_iter = perturb_hidden(
                past,
                model,
                last,
                unpert_past=unpert_past,
                unpert_logits=unpert_logits,
                accumulated_hidden=accumulated_hidden,
                grad_norms=grad_norms,
                stepsize=current_stepsize,
                loss_type=loss_type,
                num_iterations=num_iterations,
                horizon_length=horizon_length,
                window_length=window_length,
                decay=decay,
                gamma=gamma,
                kl_scale=kl_scale,
                device=device,
                verbosity_level=verbosity_level,
                output_so_far=output_so_far,
                tokenizer=tokenizer,
                bert_tokenizer=bert_tokenizer,
                last_response=last_response,
                reverse_model=reverse_model,
                context=context_t,
                detect_model=detect_model
            )
            loss_in_time.append(loss_this_iter)
        else:
            pert_past = past
    return pert_past


def generate_text_pplm(
        model,
        tokenizer,
        context=None,
        past=None,
        device="cuda",
        perturb=True,
        loss_type=0,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        bert_tokenizer=None,
        last_response=None,
        reverse_model=None,
        detect_model=None
):
    output_so_far = None
    system_response =None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t
    
    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    last = output_so_far[:, -1:]

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)

    for i in range_func:

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
        unpert_last_hidden = unpert_all_hidden[-1]

        
        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_hidden(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbosity_level=verbosity_level,
                    output_so_far=output_so_far,
                    tokenizer=tokenizer,
                    bert_tokenizer=bert_tokenizer,
                    last_response=system_response,
                    reverse_model=reverse_model,
                    context=context_t,
                    detect_model=detect_model
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)
        
        if last.tolist()[0][0] == 50256:
            # ***avoid system_response = None***
            if system_response is None:
                system_response = output_so_far
            break
        # bert = 30522 & GPT = 50256
        elif last.tolist()[0][0] <= 50257:
            output_so_far = (
                last if output_so_far is None
                else torch.cat((output_so_far, last), dim=1)
            )
            system_response = (
                last if system_response is None
                else torch.cat((system_response, last), dim=1)
            )
        else:
            print(last.tolist()[0][0])
            name = input('pause of word_id out of 50256: ')
            print('continue: ', name)

        if verbosity_level >= REGULAR:
            print(tokenizer.decode(output_so_far.tolist()[0]))

    return output_so_far, unpert_discrim_loss, loss_in_time, system_response


def run_pplm_example(
        pretrained_model="gpt2-medium",
        cond_text="",
        uncond=False,
        num_samples=1,
        bag_of_words=None,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        no_cuda=False,
        colorama=False,
        verbosity='regular',
        finetune_generation_model=None,
        emotional_valence_config=None,
        emotional_valence_model=None,
        information_relevance_config=None,
        information_relevance_model=None,
        out_dir=None,
        for_test_run=None
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
       
    pretrained_model = "microsoft/DialoGPT-medium"
    config = AutoConfig.from_pretrained(pretrained_model, output_hidden_states=True)
    model = AutoModelForCausalLM.from_config(config)

    # load pretrained model
    if finetune_generation_model != None:
        weights = torch.load(finetune_generation_model)
    
        # fix misused key value
        weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
        weights.pop("lm_head.decoder.weight", None)
        model.load_state_dict(weights, strict=False)

    model.to(device)
    model.eval()

    print("check for == some weight ==")

    if emotional_valence_config != None and emotional_valence_model != None:
        # Load a trained model and config that you have fine-tuned for pred next user utterance valence
        config = BertConfig(emotional_valence_config)
        detect_model = SequenceClassification(config, num_labels=1).to(device)
        detect_model.load_state_dict(torch.load(emotional_valence_model))
        detect_model.eval()
    else: 
        detect_model = None

    if information_relevance_config != None and information_relevance_model != None:
        cfg = GPT2Config.from_json_file(information_relevance_config)
        weights = torch.load(information_relevance_model)
        # fix misused key value
        weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
        weights.pop("lm_head.decoder.weight", None)

        reverse_model=GPT2LMHeadModel(cfg)
        reverse_model.load_state_dict(weights, strict=False)
        reverse_model.to(device)
        reverse_model.eval()
    else: 
        reverse_model = None

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    bert_tokenizer = BertTokenizer(vocab_file='./vocab/bert-base-uncased-vocab.txt')

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    path_in = './data/'
    
    file_in = open(path_in + 'test_user_utter.txt', 'r', encoding='utf-8')
    print("===dir: ", out_dir, type(out_dir))
    global file_loss
    if out_dir != None:
        if not os.path.exists('output/{}'.format(os.path.dirname(out_dir))):
            print("===create dir===")
            os.makedirs('output/{}'.format(os.path.dirname(out_dir)))
        file_pert = open('output/' + out_dir + '/perturb_all.txt', 'w+', encoding='utf-8')
        file_loss = open('output/' + out_dir + '/loss_record_perturb_all.txt', 'w+', encoding='utf-8')
    else:
        file_pert = open('output/perturb_all.txt', 'w+', encoding='utf-8')
        file_loss = open('output/loss_record_perturb_all.txt', 'w+', encoding='utf-8')

    # # # === begin time ====
    begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    sentence_count = 0
    for line in file_in.readlines():
        line = line.strip() + '<|endoftext|>'
        tokenized_cond_text = tokenizer.encode(
            # tokenizer.bos_token + raw_text,
            line,
            add_special_tokens=False
        )

        # # early stop for test
        if for_test_run != None:
            if sentence_count == 5:
                break

        sentence_count += 1
        if sentence_count %100 == 0:
            print("===" + str(sentence_count) + "===")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            
        if verbosity_level >= REGULAR:
            print("= Prefix of sentence =")
            print(tokenizer.decode(tokenized_cond_text))
            print()

        # generate unperturbed and perturbed texts

        # full_text_generation returns:
        # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time ## full_text_generation -> full_text_generation_sentence_level
        unpert_gen_tok_text, pert_gen_tok_texts, _, _, original_response, pert_response = full_text_generation_sentence_level(
            model=model,
            tokenizer=tokenizer,
            context=tokenized_cond_text,
            device=device,
            num_samples=num_samples,
            bag_of_words=bag_of_words,
            discrim=discrim,
            # class_label=class_label,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level,
            bert_tokenizer=bert_tokenizer,
            reverse_model=reverse_model,
            detect_model=detect_model
        )

        # untokenize unperturbed text
        unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])
        
        if verbosity_level >= REGULAR:
            print("=" * 80)
            print("= Unperturbed generated text =")
            print(unpert_gen_text)
            print()

        dec_sentence = tokenizer.decode(pert_response.tolist()[0])
        dec_sentence = dec_sentence.strip()
        file_pert.write(dec_sentence + '\n')
    
    # # # === finish time ===
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # file_loss.write("begin time: " + begin_time + "\t")
    # file_loss.write("finish time: " + finish_time + "\n")
    print(begin_time)
    print(finish_time)

    struct_time = time.strptime(begin_time, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
    time_stamp_begin = int(time.mktime(struct_time)) # 轉成時間戳


    struct_time = time.strptime(finish_time, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
    time_stamp_finish = int(time.mktime(struct_time)) # 轉成時間戳

    # if file_loss != None:
    #     file_loss.write(str(time_stamp_finish - time_stamp_begin)+' \n')
    print("total time(second): ", time_stamp_finish - time_stamp_begin)

    file_in.close()
    file_pert.close()
    file_loss.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake",
        help="Prefix texts to condition on"
    )
    parser.add_argument(
        "--uncond", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None,
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None,
                        help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.18)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")
    parser.add_argument("--finetune_generation_model", type=str, default=None)
    parser.add_argument("--emotional_valence_config", type=str, default=None)
    parser.add_argument("--emotional_valence_model", type=str, default=None)
    parser.add_argument("--information_relevance_config", type=str, default=None)
    parser.add_argument("--information_relevance_model", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--for_test_run", type=str, default=None)
    
    args = parser.parse_args()
    run_pplm_example(**vars(args))
