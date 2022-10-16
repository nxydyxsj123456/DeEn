import numpy as np
import  re
import  pandas as pd
from model import *
import  torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
def encoder(training_path, val_path, test_path):
    # Get training data from files
    training_data = []
    decoder_input = []
    decoder_output = []
    with open(training_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # Separate Chinese and English, line[0] is the English sentence and line[1] is the translated Chinese
            #line = line.split(' 	')
            #get http://localhost:8080/tienda1/index.jsp
            line = line.strip();
            line =line.replace('/', ' ');
            line =line.replace(':', ' ');
            line =line.replace('.', ' ');
            line =line.replace('&', ' ');
            line =line.replace('+', ' ');
            line =line.replace('-', ' ');
            line = line.replace('?', ' ');
            line = line.replace('=', ' ');

            line = re.sub(' +', ' ', line)

            print(line);
            decoder_input.append('[BOS] '+line)
            decoder_output.append(line+ ' [EOS]')

            line += ' [EOS]'
            line = '[BOS] ' + line
            line= line.split(' ')
            training_data.append([line,line])

    train_sz = len(training_data)
    training_data = np.array(training_data,dtype=object)


    return training_data
    # Build English one-hot encoder
    max_eng_vocabulary = 10000
    max_eng_sentence_length = 50

    # eng_vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    #     max_tokens=max_eng_vocabulary,
    #     output_mode='int',
    #     output_sequence_length=max_eng_sentence_length,
    #     pad_to_max_tokens=True
    # )
    # eng_vectorize_layer.adapt(training_data[:, 0])
    # eng_vocabulary = eng_vectorize_layer.get_vocabulary()
    # print(eng_vocabulary[:20])
    #
    # train_text = eng_vectorize_layer(training_data[:, 0])
    #
    # # Build Chinese one-hot encoder
    # max_chi_vocabulary = 10000
    # max_chi_sentence_length = 50
    #
    # chi_vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    #     max_tokens=max_chi_vocabulary,
    #     output_mode='int',
    #     output_sequence_length=max_chi_sentence_length,
    #     pad_to_max_tokens=True
    # )
    # chi_vectorize_layer.adapt(training_data[:, 1])
    # chi_vocabulary = chi_vectorize_layer.get_vocabulary()
    # print(chi_vocabulary[:20])

    # train_label = chi_vectorize_layer(training_data[:, 1])
    # decoder_input = chi_vectorize_layer(decoder_input)
    # decoder_output = chi_vectorize_layer(decoder_output)
    #
    # # Create training dataset
    # train_ds = tf.data.Dataset.from_tensor_slices((train_text, train_label))
    # train_ds = train_ds.shuffle(buffer_size=train_sz)

    # Get validation dataset from files
    # val_data = []
    # with open(val_path, encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         # Separate Chinese and English, line[0] is the English sentence and line[1] is the translated Chinese
    #         line = line.split(' 	')
    #         line[0] += '[EOS]'
    #         line[0] = '[BOS] ' + line[0]
    #         line[1] += '[EOS]'
    #         line[1] = '[BOS] ' + line[1]
    #         val_data.append(line)
    #
    # val_sz = len(val_data)
    # val_data = np.array(val_data)
    #
    # val_text = eng_vectorize_layer(val_data[:, 0])
    # val_label = chi_vectorize_layer(val_data[:, 1])

    # val_ds = tf.data.Dataset.from_tensor_slices((train_text, train_label))

    # Get test dataset from files
    # test_data = []
    # with open(test_path, encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         # Separate Chinese and English, line[0] is the English sentence and line[1] is the translated Chinese
    #         line = line.split(' 	')
    #         line[0] += '[EOS]'
    #         line[0] = '[BOS] ' + line[0]
    #         line[1] += '[EOS]'
    #         line[1] = '[BOS] ' + line[1]
    #         test_data.append(line)
    #
    # test_sz = len(test_data)
    # test_data = np.array(test_data)
    #
    # test_text = eng_vectorize_layer(test_data[:, 0])
    # test_label = chi_vectorize_layer(test_data[:, 1])

    # test_ds = tf.data.Dataset.from_tensor_slices((train_text, train_label))
    #
    # # Batch training dataset and validation dataset
    # batch_sz = 64
    # train_ds = train_ds.batch(batch_sz)
    # val_ds = val_ds.batch(batch_sz)
    # test_ds = test_ds.batch(batch_sz)
    #
    # encoder_input = train_text

    # return train_ds, val_ds, test_ds, eng_vocabulary, chi_vocabulary, encoder_input, decoder_input, decoder_output  #encoder_input50个数字号码，input带BOS的 teacher output 带EOS的loss_count


def text2tokens(word2id, text, do_lower_case=True):
    output_tokens = []
    text_list =list(text)
    for i in text_list:
        if i in word2id.keys():
            output_tokens.append(word2id[i])
    return output_tokens

training_path = './data/normal.txt'
val_path = './data/normal.txt'
test_path = './data/normal.txt'


result = encoder(training_path, val_path, test_path)
vocs =set()


dic1 = {}
dic1['PAD'] = 0
dic2 = {}
dic2[0] = 'PAD'

i = 1
for sentence in result[:,0] :
    for voc in sentence:
        if (voc not  in dic1.keys()) :
            dic1[voc] = i
            dic2[i] = voc
            i += 1

print(dic1)
print(dic2)

all=[]
maxlen=0
for sentence in result[:,0] :
    tmpsentence=[]
    for voc in sentence:
        tmpsentence.append(dic1[voc] )
    maxlen=max(maxlen,len(tmpsentence))

    while(len(tmpsentence)<60):
        tmpsentence.append(0)

    all.append(tmpsentence)

all=np.array(all)
all=np.expand_dims(all,axis=1)
all=np.concatenate([all,all], axis= 1)

print(all)
print(maxlen) #53  padding 到60



INPUT_DIM = OUTPUT_DIM =len(len(dic1))

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)
TRG_PAD_IDX = 0#TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

