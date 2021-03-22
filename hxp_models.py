import torch
import torch.nn as nn
import numpy as np
import random


class Encoder(nn.Module):
    """
        input_dim : the size/dimensionality of the one-hot vectors that will be input to the encoder.
    This is equal to the input (source) vocabulary size.源语言的词典大小。单词个数。
        emb_dim : the dimensionality of the embedding layer.
    This layer converts the one-hot vectors into dense vectors with emb_dim dimensions.词向量的维度。
        hid_dim : the dimensionality of the hidden and cell states.隐藏层维度。
        n_layers : the number of layers in the RNN.
        dropout : the amount of dropout to use. This is a regularization parameter to prevent overfitting.
    Check out this for more details about dropout.
    """
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        # nn.Embedding 参数介绍：
        # num_embeddings(python:int) – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
        # embedding_dim(python:int) – 嵌入向量的维度，即用多少维来表示一个符号。
        # padding_idx(python:int,optional) – 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，
        # 而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
        # 输入: (∗) , 包含提取的编号的任意形状的长整型张量。
        # 输出: (∗,H) , 其中 * 为输入的形状，H为embedding_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # nn.LSTM 参数介绍：
        # input_size 输入数据的特征维数，通常就是embedding_dim(词向量的维度)
        # hidden_size　LSTM中隐层的维度
        # num_layers　循环神经网络的层数
        # 输入: input,(h_0,c_0)
        # input就是shape==(seq_length,batch_size,embedding_dim)的张量
        # h_0的shape==(num_layers*num_directions,batch,hidden_size)的张量
        # 它包含了在当前这个batch_size中每个句子的初始隐藏状态
        # c_0和h_0的形状相同，它包含的是在当前这个batch_size中的每个句子的初始细胞状态。
        # h_0,c_0如果不提供，那么默认是０
        # 输出: output,(h_n,c_n):
        # output的shape==(seq_length,batch_size,num_directions*hidden_size)
        # h_n.shape==(num_directions * num_layers,batch,hidden_size)
        # c_n.shape==h_n.shape
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        # src_len表示句子的单词个数，大小不固定，又因为是一个batch进来的，所以src_len统一为这个batch里所有句子单词数的最大值
        # 这里src_len大小不固定没有问题，因为这只会影响到output的size，不影响hidden和cell的size，decoder只需要hidden和cell

        # embedded = [src len, batch size, emb dim]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        output, (hidden, cell) = self.lstm(embedded)

        return hidden, cell


class Decoder(nn.Module):
    """
        output_dim : the size/dimensionality of the one-hot vectors that will be input to the encoder.
    This is equal to the input (source) vocabulary size.目标语言词典大小。单词个数。
        emb_dim : the dimensionality of the embedding layer.
    This layer converts the one-hot vectors into dense vectors with emb_dim dimensions.词向量的维度。
        hid_dim : the dimensionality of the hidden and cell states.隐藏层维度。
        n_layers : the number of layers in the RNN.
        dropout : the amount of dropout to use. This is a regularization parameter to prevent overfitting.
    Check out this for more details about dropout.
    """
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        # input = [seq_len = 1, batch size]
        # 这里seq_len为1是因为，解码器是一个单词一个单词地预测的，前面的结果影响后面的结果
        # 在seq_seq类中解码器也是每预测一个单词调用一次decoder的forward函数
        input = input.unsqueeze(0)

        # embedded size=[1, batch_size, emb_dim]
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        output, (hidden, cell) = self.lstm(embedded)

        # prediction size=[batch_size, output_dim]
        prediction = self.fc(output.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs [trg_len, batch_size, trg_vocab_size]
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]d
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # hidden = [n layers, batch size, hid dim]
            # cell = [n layers, batch size, hid dim]
            # output size=[batch_size, output_dim=trg_vocab_size]
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teach = random.random() < teacher_forcing_ratio

            # get the highest predicted token (index) from our predictions
            # top1 size=[batch_size]
            # argmax 返回最大值的索引而不是最大值
            # 针对二维数组:
            # axis = 0, 返回数组中每一列最大值所在行索引。
            # axis = 1, 返回数组中每一行最大值所在列索引。
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teach else top1
        return outputs
