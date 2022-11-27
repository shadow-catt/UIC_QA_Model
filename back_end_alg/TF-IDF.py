import jieba
import pandas as pd
import numpy as np
from gensim import corpora, models, similarities
from termcolor import colored
import time
def read_and_split_the_excel(path):
    """
    :func: 根据xlsx文件获取问题list和答案list（需要更新openyxl）
    :param path: 文件路径
    :return: 问题list，答案list
    """
    # 读取文件
    df1 = pd.read_excel(path)
    # 分开
    question_list = df1.iloc[:,0].tolist()
    answer_list = df1.iloc[:,1].tolist()
    # 返回
    return question_list,answer_list

# # 测试read_and_split_the_excel
# question_list,answer_list = read_and_split_the_excel("./dataset/CN_QA_dataset_all.xlsx")
# print(question_list[:3])

# 导入停用词表
def obtain_stop_word(path):
    """
    :func: 获取stop_word
    :param path: 文件路径
    :return: 返回stop_word list
    """
    stop_words = [line.strip() for line in open(path).readlines()]
    stop_words.extend([""," "])
    return stop_words

# # obtain stop word 代码测试
# # 使用的是cn_stopwords，在kaggle搜索哈工大第一个
# path = './dataset/cn_stopwords.txt'
# stop_words = obtain_stop_word(path)
# print(stop_words[:25])

def cn_stop_word_rm(sentence, stop_words):
    """
    :func: 将输入的句子分词并且移除stopword，返回list
    :param stop_words: 需要移除的stopword（用的是cn_stopwords）
        eg:
                ['$', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                '9', '?', '_', '“', '”', '、', '。','《', '》', '一',
                '一些', '一何', '一切', '一则', '一方面', '一旦', '一来']
    :param sentence: 句子
        eg:
            "今天我想摆烂，你能拿我咋办，摸鱼我说了算"
    :return: 返回分词后的token list
    """
    # split the sentence
    word_tokens = list(jieba.cut_for_search(sentence))

    # remove stop words
    query = [w.lower() for w in word_tokens if not w in stop_words]
    #     print(query)
    #     question_list[index] = ' '.join(line for line in query)
    return query

# # 测试cn_stop_word_rm
# sentence = "我们认为，一些关键问题就是所谓问题的关键，所以问题的关键在于我们如何把握关键问题，这个是我们任务的关键"
# path = './dataset/cn_stopwords.txt'
# stop_words = obtain_stop_word(path)
# query = cn_stop_word_rm(sentence,stop_words)
# print(query)

def generate_question_t_list(question_list, stop_words):
    """
    :func: 将输入的问句分词逐个转为token list

    :param question_list: 句子列表
        eg:
            ["今天我想摆烂"，
            "你能拿我咋办"，
            "摸鱼我说了算"]

    :param stop_words: 需要移除的stopword（用的是cn_stopwords）
        eg:
                ['$', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                '9', '?', '_', '“', '”', '、', '。','《', '》', '一',
                '一些', '一何', '一切', '一则', '一方面', '一旦', '一来']

    :return: 返回question_list每句分词后的token list
    """

    # transfer the question list into the token form
    question_token_list = ['' for i in range(len(question_list))]

    for index in range(len(question_list)):
        # split the sentence
        question_token_list[index] = cn_stop_word_rm(question_list[index], stop_words)
    # return the token list
    return question_token_list

# # 测试generate_question_t_list
# question_list,answer_list = read_and_split_the_excel("./dataset/CN_QA_dataset_all.xlsx")
# path = './dataset/cn_stopwords.txt'
# stop_words = obtain_stop_word(path)
# question_token_list = generate_question_t_list(question_list, stop_words)
# print(question_token_list[:4])


def similarity_cn(query, dictionary, tfidf, corpus_tfidf, topk=3, threshold=0.7, all_score_without_rank=0):
    """
    :func: 计算问题与知识库中问题的相似度
    :param Corp: 分词后的问题
        eg:
                [['UIC', '学校', '办学', '性质'],
                 ['学校', '现在', '在校', '在校生'],
                 ['UIC', '全称'],
                 ['北师', '北师港', '浸大', '全称']]
    :param query: 分词后的问题
        eg:
                ['UIC', '全称', '名字']
    :return: 返回
        if_valid: 最匹配的答案的相似度是否超过了threshold
        max_loc: 前topk个最匹配的答案所在的index
        sims: 每一个问题与查找问题的相似度，依据index来的
    """

    #     print(corpus_tfidf)
    # # 得到TF-IDF值
    #     for temp in corpus_tfidf:
    #         print(temp)

    vec_bow = dictionary.doc2bow(query)
    vec_tfidf = tfidf[vec_bow]
    #     print(vec_tfidf)

    index = similarities.MatrixSimilarity(corpus_tfidf)
    #     print(index)

    sims = index[vec_tfidf]
    #     print(sims)

    if all_score_without_rank:
        return sims
    else:
        max_loc = np.argsort(sims)[::-1][:topk]
        #     print(np.argsort(sims)[::-1])

        #     top_max_sim = sims[max_loc]
        #     print(top_max_sim)

        # if the score is larger than the threshold
        if sims[max_loc[0]] < threshold:
            if_valid = 0
        else:
            if_valid = 1

        return if_valid, max_loc, sims



# # 测试，用query的问句去问系统，这里面返回前三个的相似度
# # generate question_t_list
# question_list,answer_list = read_and_split_the_excel("./dataset/CN_QA_dataset_all.xlsx")
# path = './dataset/cn_stopwords.txt'
# stop_words = obtain_stop_word(path)
# question_token_list = generate_question_t_list(question_list, stop_words)
#
# Corp = question_token_list
# query = ['uic', '全称', '名字']
# if_vaild, max_loc, top_max_sim = similarity_cn(Corp, query)
# print(if_vaild)
# print(max_loc)
# print(top_max_sim[:3])

def TF_IDF_prepared(data_file_path, stopword_file_path):
    # 准备TF-IDF回答过程中需要的材料
    # read the file
    question_list, answer_list = read_and_split_the_excel(data_file_path)

    # stop words list
    stop_words = obtain_stop_word(stopword_file_path)

    # generate question token list
    question_token_list = generate_question_t_list(question_list, stop_words)

    # 建立词典
    dictionary = corpora.Dictionary(question_token_list)

    # 基于词典，将分词列表集转换成稀疏向量集，即语料库
    corpus = [dictionary.doc2bow(text) for text in question_token_list]

    # 训练TF-IDF模型，传入语料库进行训练
    tfidf = models.TfidfModel(corpus)

    # 用训练好的TF-IDF模型处理被检索文本，即语料库
    corpus_tfidf = tfidf[corpus]

    return question_list, question_token_list, answer_list, stop_words,dictionary, tfidf, corpus_tfidf

def TF_IDF_reply(query, question_list, answer_list, stop_words, topk_TFIDF,threshold_TFIDF, dictionary, tfidf, corpus_tfidf):
    start_time = time.time()
    # 对查询的问题进行处理
    query_processed = cn_stop_word_rm(query, stop_words)

    # 得到问题（答案）所对应的行索引
    if_valid, topk_idx_TF, score_TF = similarity_cn(query_processed, dictionary, tfidf, corpus_tfidf, topk_TFIDF, threshold_TFIDF, all_score_without_rank=0)
    end_time = time.time()
    wasting_time = end_time-start_time
    print("Costing time for getting answer:", wasting_time)

    #回答答案，可以注释
    print('top few questions(TFIDF: %d) similar to "%s"' % (topk_TFIDF, colored(query, 'green')))
    print("The best similarity for TF-IDF is:", score_TF[topk_idx_TF[0]])

    for idx in topk_idx_TF:
        print('TF-IDF; %s\t%s' % (colored('%.4f' % score_TF[idx], 'cyan'), colored(question_list[idx], 'yellow')))

    # get the answer
    if if_valid:
        print(answer_list[topk_idx_TF[0]])

    return if_valid, topk_idx_TF, score_TF
# 下面为主函数
def TF_IDF_QA():
    # file storing the data
    print("数据准备中")
    data_file_path = "./dataset/CN_QA_dataset_all.xlsx"
    stopword_file_path = './dataset/cn_stopwords.txt'
    # 显示前几个答案
    topk_TFIDF = 3
    # 超过多少相似度才算合格
    threshold_TFIDF = 0.7
    # data prepared
    question_list, question_token_list, answer_list, stop_words, dictionary, \
        tfidf, corpus_tfidf = TF_IDF_prepared(data_file_path, stopword_file_path)
    print("准备完毕")
    # obtain the question

    while True:
        # 读取数据
        query = input('请输入问题（输入quit退出）: ')
        if query == "quit":
            break

        TF_IDF_reply(query, question_list, answer_list, stop_words, topk_TFIDF,
                     threshold_TFIDF, dictionary, tfidf, corpus_tfidf)

if __name__ == '__main__':
    TF_IDF_QA()