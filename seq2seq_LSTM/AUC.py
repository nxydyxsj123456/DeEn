import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

#import spacy
import numpy as np

import random
import math
import time
import  re

from dataset import MyDataset
from  model import *
from seq2seq_LSTM.model_LSTM import Seq2SeqX


def loaddata(path):
    data = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # get http://localhost:8080/tienda1/index.jsp
            line = line.strip();
            line = line.replace('/', ' ');
            line = line.replace(':', ' ');
            line = line.replace('.', ' ');
            line = line.replace('&', ' ');
            line = line.replace('+', ' ');
            line = line.replace('-', ' ');
            line = line.replace('?', ' ');
            line = line.replace('=', ' ');
            line = re.sub(' +', ' ', line)
            print(line);
            line += ' [EOS]'
            line = '[BOS] ' + line
            line = line.split(' ')
            data.append(line)

    data = np.array(data, dtype=object)
    return data

def getToken(text_data):

    text2tokensdic = {}
    text2tokensdic['PAD'] = 0

    tokens2textdic = {}
    tokens2textdic[0] = 'PAD'

    print(type(text2tokensdic))
    print(text2tokensdic.keys())

    i = 1
    for sentence in text_data:
        for voc in sentence:
            if (voc not in text2tokensdic.keys()):
                text2tokensdic[voc] = i
                tokens2textdic[i] = voc

                i += 1
    text2tokensdic['<UNK>'] = i
    tokens2textdic[i] = '<UNK>'

    print(text2tokensdic)
    print(tokens2textdic)



    return text2tokensdic,tokens2textdic

def tokenText(training_data,text2tokensdic) :
    Tokened_text = []
    maxlen = 0
    for sentence in training_data:
        tmpsentence = []
        for voc in sentence:
            if (voc not in text2tokensdic.keys()):
                tmpsentence.append(len(text2tokensdic)-1)
            else:
                tmpsentence.append(text2tokensdic[voc])
        maxlen = max(maxlen, len(tmpsentence))
        while (len(tmpsentence) < 65):
            tmpsentence.append(0)
        Tokened_text.append(tmpsentence)

    print(maxlen)  # 53  padding 到60
    Tokened_text = np.array(Tokened_text)

    #print(Tokened_text)
    return  Tokened_text
def untokenText(output_data,tokens2textdic) :
    text = []
    for token in output_data:

        if (token.item() not in tokens2textdic.keys()):
            text.append('<UNK>')
        else:
            text.append(tokens2textdic[token.item()])
    return  text

def text2tokens(word2id, text, do_lower_case=True):
    output_tokens = []
    text_list =list(text)
    for i in text_list:
        if i in word2id.keys():
            output_tokens.append(word2id[i])
    return output_tokens

normal_path = '../data/normal.txt'
val_path = '../data/normal.txt'
anomalous_path = '../data/anomalous.txt'


normal_data=loaddata(normal_path)

anomalous_data=loaddata(anomalous_path)

text2tokensdic,tokens2textdic=getToken(normal_data)


Tokened_normal=tokenText(normal_data, text2tokensdic)

Tokened_anomalous=tokenText(anomalous_data, text2tokensdic)



INPUT_DIM = OUTPUT_DIM =len(text2tokensdic)

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
from model_LSTM import  Seq2SeqX,Encoder,Decoder,Attention
attn = Attention(ENC_HID_DIM, DEC_HID_DIM)

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

#from model_LSTM import  Seq2SeqX
model = Seq2SeqX(enc, dec, device).to(device)
TRG_PAD_IDX = 0#TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(model, iterator, optimizer, criterion):
    model.train()  # 进入训练模式
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch[0].transpose(1,0)
        trg = batch[1].transpose(1,0).long()  # trg = [trg_len, batch_size]


        # pred = [trg_len, batch_size, pred_dim]
        pred = model(src, trg)

        pred_dim = pred.shape[-1]

        # trg = [(trg len - 1) * batch size]
        # pred = [(trg len - 1) * batch size, pred_dim]
        trg = trg[1:].contiguous().view(-1)
        pred = pred[1:].view(-1, pred_dim)

        loss = criterion(pred, trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


"""...and the evaluation loop, remembering to set the model to `eval` mode and turn off teaching forcing."""


def evaluate(model, iterator, criterion):
    scores = []

    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            print(i)
            src = batch[0].transpose(1, 0)
            trg = batch[1].transpose(1, 0).long()  # trg = [trg_len, batch_size]

            test_sentence=[]

            for j in range(len(batch[0])):
                #print(max(batch[0][j]))
                tmpsentence=untokenText(batch[0][j], tokens2textdic)

                tmpsentence = [i for n, i in enumerate(tmpsentence) if i not in ['[BOS]', 'PAD', '[EOS]']]

                test_sentence .append(tmpsentence)



            #print(test_sentence)

            # output = [trg_len, batch_size, output_dim]
            output = model(src, trg, 0)  # turn off teacher forcing
            tmps=output[:, 0, :]
            output_token = torch.argmax(output,dim=2)

            output_text = []
            for j in range(output_token.shape[1]):
                tmpsentence=untokenText(output_token[:,j],tokens2textdic)

                tmpsentence = [i for n, i in enumerate(tmpsentence) if i not in ['[BOS]','PAD','[EOS]']]

                output_text.append(tmpsentence)
            #print(output_text)

            # reference = [['this', 'is', 'a', 'test', 'ojbk', 'ojbk']]
            # candidate = ['this', 'is', 'a', 'test']

            for j in range(len(output_text)):
                scores.append(sentence_bleu([test_sentence[j]], output_text[j]))


            # print(test_sentence[0])
            # print(output_text[0])
            # print(scores[0])


            output_dim = output.shape[-1]

            # trg = [(trg_len - 1) * batch_size]
            # output = [(trg_len - 1) * batch_size, output_dim]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator),scores


"""Finally, define a timing function."""


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


"""Then, we train our model, saving the parameters that give us the best validation loss."""

best_valid_loss = float('inf')
trainingXdataset=MyDataset(Tokened_normal)

testXdataset=MyDataset(Tokened_anomalous)


train_iterator = DataLoader(trainingXdataset,64)

test_iterator = DataLoader(testXdataset,64)


"""Finally, we test the model on the test set using these "best" parameters."""

model.load_state_dict(torch.load('./9model.pt'))

test_loss,scores1 = evaluate(model, test_iterator, criterion)
train_loss,scores2 = evaluate(model, train_iterator, criterion)


print(max(scores1),min(scores2))

# 保存
import numpy as np
scores1=np.array(scores1)
np.save('scores_bad.npy',scores1)   # 保存为.npy格式

scores2=np.array(scores2)
np.save('scores_good.npy',scores2)   # 保存为.npy格式
# 读取


#print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

"""We've improved on the previous model, but this came at the cost of doubling the training time.

In the next notebook, we'll be using the same architecture but using a few tricks that are applicable to all RNN architectures - packed padded sequences and masking. We'll also implement code which will allow us to look at what words in the input the RNN is paying attention to when decoding the output.
"""














import  numpy as np
from sklearn import  metrics

# scores_good=np.load('scores_good.npy')
# scores_bad=np.load('scores_bad.npy')


label1=np.zeros(len(scores2))
label2=np.ones(len(scores1))

label=np.append(label1,label2)
pred=np.append(scores2,scores1)

fpr,tpr,thresholds = metrics.roc_curve(label,pred,pos_label=0)

auc= metrics.auc(fpr,tpr)

print(auc)