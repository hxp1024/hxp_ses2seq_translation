from hxp_models import Encoder, Decoder, Seq2Seq
from hxp_data_preprocessing import data_preprocessing
import hxp_config as config
import torch
import torch.nn as nn
import torch.optim as optim
import spacy

import numpy as np


# We initialize weights in PyTorch by creating a function which we apply to our model.
# When using apply, the init_weights function will be called on every module and sub-module within our model.
# For each module we loop through all of the parameters and
# sample them from a uniform distribution with nn.init.uniform_.
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def train(model, iterator, optimizer, criterion, clip, model_path):
    model.train()
    sum_loss = 0.0

    for i, data_i in enumerate(iterator):
        # src size=[src_len, batch]
        # trg size=[trg_len, batch]
        src = data_i.src
        trg = data_i.trg

        optimizer.zero_grad()

        # output size=[trg_len, batch_size, trg_vocab_size]
        output = model(src, trg)

        # output_dim = trg_vocab_size
        output_dim = output.shape[-1]

        # [trg_len, batch_size, trg_vocab_size] -> [trg_len*batch_size, trg_vocab_size]
        output = output[1:].view(-1, output_dim)
        # [trg_len, batch] -> [trg_len*batch]
        trg = trg[1:].view(-1)

        loss_i = criterion(output, trg)

        loss_i.backward()

        # 梯度裁剪（Clipping Gradient）: 既然在BP过程中会产生梯度消失/爆炸（就是偏导无限接近0，导致长时记忆无法更新）
        # 那么最简单粗暴的方法: 设定阈值。当梯度小于/大于阈值时，更新的梯度为阈值
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        sum_loss += loss_i.item()

        print(i, loss_i.item())
        # if i % 10 == 0:
        #     torch.save(model.state_dict(), model_path)

    return sum_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    sum_loss = 0

    with torch.no_grad():
        for i, data_i in enumerate(iterator):
            # src size=[src_len, batch]
            # trg size=[trg_len, batch]
            src = data_i.src
            trg = data_i.trg

            # output size=[trg_len, batch_size, trg_vocab_size]
            output = model(src, trg)

            # output_dim = trg_vocab_size
            output_dim = output.shape[-1]

            # [trg_len, batch_size, trg_vocab_size] -> [trg_len*batch_size, trg_vocab_size]
            output = output[1:].view(-1, output_dim)
            # [trg_len, batch] -> [trg_len*batch]
            trg = trg[1:].view(-1)

            loss_i = criterion(output, trg)

            sum_loss += loss_i.item()
    return sum_loss / len(iterator)


def train_and_save_model(model_path):
    # TRG 是英文单词编码的字典，即：
    # TRG.vocab.itos 是 int to string 的一个字典
    # TRG.vocab.stoi 是 string to int 的一个字典
    # SRC 同理，是法语单词编码的字典
    SRC, TRG, device, train_iterator, valid_iterator, test_iterator = data_preprocessing()
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    enc = Encoder(INPUT_DIM, config.ENC_EMB_DIM, config.HID_DIM, config.N_LAYERS, config.ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, config.DEC_EMB_DIM, config.HID_DIM, config.N_LAYERS, config.DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device, 0.5).to(device)

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    best_loss = float('inf')

    for i in range(config.N_EPOCHS):

        train_loss = train(model, train_iterator, optimizer, criterion, config.CLIP, model_path)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), model_path)


def predict_sentence(sentence, model_path):
    SRC, TRG, device, train_iterator, valid_iterator, test_iterator = data_preprocessing()
    device = torch.device('cpu')
    # SRC = SRC.to(device)
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    enc = Encoder(INPUT_DIM, config.ENC_EMB_DIM, config.HID_DIM, config.N_LAYERS, config.ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, config.DEC_EMB_DIM, config.HID_DIM, config.N_LAYERS, config.DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device, 0).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    spacy_de = spacy.load('de_core_news_sm')
    sentence = [tok.text for tok in spacy_de.tokenizer(sentence)][::-1]
    print(sentence)
    sentence = [SRC.vocab.stoi[w] for w in sentence]
    print(sentence)
    sentence = torch.from_numpy(np.array(sentence))
    print(sentence)
    sentence = sentence.unsqueeze(1)
    print(sentence)
    # sentence = sentence.to(device)
    src = sentence
    trg = sentence
    output = model(src, trg)
    output = output.squeeze(1)
    output = output.argmax(1)
    output = [TRG.vocab.itos[i] for i in output]
    print(src, output)


if __name__ == '__main__':
    # train_and_save_model(config.model_path)
    sentence = 'zwei junge weiße männer sind im freien in der nähe vieler büsche.'
    predict_sentence(sentence, config.model_path)
