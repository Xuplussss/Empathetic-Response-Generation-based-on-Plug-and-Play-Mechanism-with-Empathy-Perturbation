''' 
command: 
Transformer:
    python3 bleu.py -pred_file pred/fold0/pure/template/w/outside/filled.1.0.20.32.768.3072.8.3.txt
Conditional Transformer:
    python3 bleu.py -pred_file pred/fold0/conOnly_epa/template/w_test2/outside/filled.1.0.20.32.768.3072.8.3.txt
Conditional Transformer w R/F + Emo Class Loss:
    python3 bleu.py -pred_file pred/fold0/ac_epa/template/w_bertEmbed/outside/filled.1.1.20.32.768.3072.8.3.txt
Conditional Transformer w R/F Loss:
    python3 bleu.py -pred_file pred/fold0/ac_epa/template/w_bertEmbed_adv/outside/filled.1.1.20.32.768.3072.8.3.txt
Conditional Transformer w Emo Class Loss:
    python3 bleu.py -pred_file pred/fold0/ac_epa/template/w_bertEmbed_aux/outside/filled.1.1.20.32.768.3072.8.3.txt
'''

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import nltk
import sys, os
import re
import argparse
import numpy as np

nltk.download('punkt')
from nltk.tokenize import word_tokenize
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=20) 
    parser.add_argument('-batch_size', type=int, default=32)

    # for Transformer
    parser.add_argument('-d_model', type=int, default=768)
    parser.add_argument('-d_inner_hid', type=int, default=3072)
    parser.add_argument('-d_k', type=int, default=96)
    parser.add_argument('-d_v', type=int, default=96)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=3)

    parser.add_argument('-format', type=str) 
    parser.add_argument('-mode', type=str) 

    parser.add_argument('-pred_file', type=str, required=True) # preficted result file
    parser.add_argument('-label_file', type=str, required=True) # preficted result file

    opt = parser.parse_args()

    # opt.folder_name = opt.pred_file.split('/')[2]
    # opt.format = opt.pred_file.split('/')[3] # end2end/template
    # opt.mode = opt.pred_file.split('/')[4]

    # open a new file for bleu output result
    if not os.path.exists('bleu/{}'.format(os.path.dirname(opt.pred_file))):
        os.makedirs('bleu/{}'.format(os.path.dirname(opt.pred_file)))

    out_file = open('bleu/{}'.format(opt.pred_file), 'w+', encoding='utf-8')
    out_file.close()

    bleus_folds = []
    for fold_num in range(1):
        smoothing = SmoothingFunction()
        bleus = {}
        score = 0
        count = 0
        # predict
        preds = []
        with open(opt.pred_file, 'r', encoding='utf-8') as fpred:
            for line in fpred:
                # line = line.strip()
                line = line.replace('[SEP]','').strip()
                # words = line.split(' ')
                words = word_tokenize(line)
                # words = list(line) # character
                preds.append(words)
        
        # validate
        labels = []
        with open(opt.label_file, 'r', encoding='utf-8') as fvald:
            for line in fvald:
                line = line.strip()
                # words = line.split(' ')
                words = word_tokenize(line)
                # words = list(line)
                labels.append([words])

        # calculate bleu score
        for v, p in zip(labels, preds):
            count += 1
            # print(v)
            # print(p)
            # print(sentence_bleu(v, p, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method2))
            score += sentence_bleu(v, p, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method2)
        print(opt.pred_file)
        print('Bleu score:', (score / count) * 100)
        bleus[re.sub(r'fold\d', 'fold{}'.format(fold_num), opt.pred_file)] = (score / count) * 100

        # write bleus into file
        with open('bleu/{}'.format(opt.pred_file), 'a', encoding='utf-8') as f:
            for k, v in bleus.items():
                f.write('fold{}\t{}\t{}\n'.format(fold_num, k, v))

        bleus_folds += list(bleus.values())
    
    print(bleus_folds)
    # with open('bleu/{}'.format(opt.pred_file), 'a', encoding='utf-8') as f:
    #     f.write('average:{}+-{}\n'.format(np.mean(bleus_folds), np.std(bleus_folds)))

if __name__ == '__main__':
    main()