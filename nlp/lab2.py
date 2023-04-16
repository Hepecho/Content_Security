import gensim
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.text import TextCollection
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

Punctuation = ['~', '`', '``', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '-', '=', '{', '}', '|',
               '[', ']', '\\', ':', '\"', ';', '\'', '<', '>', '?', ',', '.', '/']  # 标点符号表


def read_data(path):
    data = pd.read_csv(path)
    # print(type(data['Job Description']))
    data = data['Job Description'].tolist()
    sentences = [str(sentence) for sentence in data]
    # sys.exit()
    return sentences


def save_data(data, path):
    ans = pd.DataFrame(data=data, columns=['id', 'keywords'])
    ans.to_csv(path)


def split_word(text):
    """
    对文本进行分词、去标点符号、去停用词
    :param text: 文本库中的一段文本
    :return: text_after_lemmatize: 文本经过处理后的得到的词表
    """
    token_text = word_tokenize(text.lower())  # 将文本转为小写（也可以不转），并使用nltk对其进行分词
    text_without_punc = [word for word in token_text if word not in Punctuation]  # 去除标点符号项
    stops = set(stopwords.words("english"))  # 加载NLTK英文停用词表
    text_without_deac = [word for word in text_without_punc if word not in stops]  # 去停用词
    return text_without_deac


def clean_data(data):
    words_list = []
    sentence_list = []
    for i, sentence in enumerate(data):
        sentence = split_word(sentence)
        sentence_list.append(" ".join(sentence))
        words_list.append(sentence)
    return words_list, sentence_list


def TF_IDF(corpus, words):
    """
    计算文本中每个单词的TF-IDF值
    :param corpus: 语料库
    :param all_words: 目标文本使用上一步的split_word(text)函数处理得到的词表
    :return: tf_idf_dict: 文本中每一个单词及其TF-IDF值构成的字典
    """
    tf_idf_dict = {}
    for word in words:
        tf_idf_dict[word] = corpus.tf_idf(word, words)  # 使用nltk自带的TF-IDF函数对文本中每个词计算其TF-IDF值
    return tf_idf_dict


if __name__ == '__main__':
    data_path = 'JobDataAnalyst.csv'
    keyword_num = 20

    print("preprocess...")
    raw_data = read_data(data_path)
    words_list, sentence_list = clean_data(raw_data)
    corpus = TextCollection(words_list)  # 构建语料库
    print("preprocess successfully")

    print("tf_idf...")
    tf_idf_keywords = []

    for i, words in enumerate(words_list):
        tf_idf_dict = TF_IDF(corpus, words)
        # print(tf_idf_dict)
        tf_idf_ = sorted(tf_idf_dict.items(), key=lambda s: s[1], reverse=True)
        # print(str(i) + ":")

        if keyword_num > len(tf_idf_):
            tf_idf_keywords.append([i, tf_idf_])
        else:
            tf_idf_keywords.append([i, dict(tf_idf_[:keyword_num+1])])
    save_data(tf_idf_keywords, 'tf_idf_keywords.csv')
    print("tf_idf end")

    print("page_rank...")
    page_rank_keywords = []

    for i, sentence in enumerate(sentence_list):
        gensim_kw = gensim.summarization.keywords(sentence, words=keyword_num, split=True, scores=True)
        page_rank_keywords.append([i, dict(gensim_kw)])
    save_data(page_rank_keywords, 'page_rank_keywords')
    print("page_rank end")
