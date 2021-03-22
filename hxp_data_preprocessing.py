import random
import numpy as np
import torch
import spacy
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator


# 使用spacy创建分词器（tokenizers），分词器的作用是将一个句子转换为组成该句子的单个符号列表
# 例如。“good morning!”变成了“good”，“morning”和“!”
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


# 创建分词器函数以便传递给TorchText
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    # spacy_de = spacy.load('de_core_news_sm')
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    # spacy_en = spacy.load('en_core_web_sm')
    return [tok.text for tok in spacy_en.tokenizer(text)]


def data_preprocessing():
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # import de_core_news_sm, en_core_web_sm
    # spacy_de = de_core_news_sm.load()
    # spacy_en = en_core_web_sm.load()
    # spacy_de = spacy.load('de_core_news_sm')
    # spacy_en = spacy.load('en_core_web_sm')

    # Field对象 :指定要如何处理某个字段，比如指定分词方法，是否转成小写，起始字符，结束字符，补全字符以及词典等。
    # 我们创建SRC和TRG两个Field对象，tokenize为我们刚才定义的分词器函数
    # 在每句话的开头加入字符SOS，结尾加入字符EOS，将所有单词转换为小写。
    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    # splits方法可以同时加载训练集，验证集和测试集，
    # 参数exts指定使用哪种语言作为源语言和目标语言，fileds指定定义好的Field类
    train_data, valid_data, test_data = Multi30k.splits(
        exts=('.de', '.en'),
        fields=(SRC, TRG))

    # print(f"Number of training examples: {len(train_data.examples)}")
    # print(f"Number of validation examples: {len(valid_data.examples)}")
    # print(f"Number of testing examples: {len(test_data.examples)}")

    # vars() 函数返回对象object的属性和属性值的字典对象。
    # print(vars(train_data.examples[0]))

    # 构建词表，即给每个单词编码，用数字表示每个单词，这样才能传入模型
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    # print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    # print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print(device)

    BATCH_SIZE = 128

    # BucketIterator：相比于标准迭代器，会将类似长度的样本当做一批来处理
    # 因为在文本处理中经常会需要将每一批样本长度补齐为当前批中最长序列的长度
    # 因此当样本长度差别较大时，使用BucketIerator可以带来填充效率的提高。
    # 除此之外，我们还可以在Field中通过fix_length参数来对样本进行截断补齐操作。

    # 当使用迭代器生成一个batch时，我们需要确保所有的源语言句子都padding到相同的长度，目标语言的句子也是。
    # 这些功能torchtext可以自动的完成，其使用了动态padding，意味着一个batch内的所有句子会pad成batch内最长的句子长度。
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)

    return SRC, TRG, device, train_iterator, valid_iterator, test_iterator


if __name__ == '__main__':
    data_preprocessing()
