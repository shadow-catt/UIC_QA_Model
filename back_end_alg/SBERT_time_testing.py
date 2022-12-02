# 安装包
# pip install -U sentence-transformers
# pip install -U transformers
# pip install openpyxl
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from copy import deepcopy
from random import randint
from termcolor import colored
import time

def read_and_split_the_excel(QA_path):
    """
    :func: 根据xlsx文件获取问题list和答案list（需要更新openyxl）
    :param QA_path: 文件路径
    :return: 问题list，答案list
    """
    # 读取文件
    df1 = pd.read_excel(QA_path)
    # 分开
    question_list = df1.iloc[:, 0].tolist()
    answer_list = df1.iloc[:, 1].tolist()
    # 返回
    return question_list, answer_list


# # 测试read_and_split_the_excel
# question_list,answer_list = read_and_split_the_excel("../input/uic-cn-admission/CN_QA_dataset_all.xlsx")
# display(question_list[:3])

def read_and_split_the_01(zero_one_path):
    """
    :func: 根据xlsx文件获取原始list和测试list和label
    :param zero_one_path: 文件路径
    :return: 问题list，答案list
    """
    # 读取文件
    df1 = pd.read_csv(zero_one_path)
    # 分开
    sen1_list = df1.iloc[:, 0].tolist()
    sen2_list = df1.iloc[:, 1].tolist()
    label_list = df1.iloc[:, 2].tolist()
    # 返回
    return sen1_list, sen2_list, label_list


# # 测试read_and_split_the_excel
# Sen1_list, Sen2_list, label_list = read_and_split_the_01("../input/01-uic-rm-dup/01_all_rm_dup.csv")
# display(Sen1_list[:3])
# display(Sen2_list[:3])
# display(label_list[:3])

def shuffle(list_):
    temp_list = deepcopy(list_)
    m = len(temp_list)
    while (m):
        m -= 1
        i = randint(0, m)
        temp_list[m], temp_list[i] = temp_list[i], temp_list[m]
        return temp_list


def obtain_shuffle_01(ori_list):
    shuffle_q_list = shuffle(ori_list)

    shuffle_label_list = [0] * len(shuffle_q_list)

    return ori_list, shuffle_q_list, shuffle_label_list


# # Test the shuffle
# question_list = ['The cat sits outside',
#       'A man is playing guitar',
#       'The new movie is awesome',
#       'The new opera is nice']
# obtain_shuffle_01(question_list)

def read_qa_and_expand_training_set(QA_path, zero_one_path):
    # get the qa_data
    question_list, answer_list = read_and_split_the_excel(QA_path)
    # get the 01_data
    Sen1_list, Sen2_list, label_list = read_and_split_the_01(zero_one_path)
    # get expand 01 data
    ori_list, shuffle_q_list, shuffle_label_list = obtain_shuffle_01(question_list)
    Sen1_list.extend(ori_list)
    Sen2_list.extend(shuffle_q_list)
    label_list.extend(shuffle_label_list)

    return question_list, answer_list, Sen1_list, Sen2_list, label_list


def SBERT_get_reply(model, query, question_list, answer_list, question_list_emb, topk_SBERT, threshold_SBERT):
    start_time = time.time()
    # prepared for queries
    queries = [query]
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    if_valid = 0

    index_ranked = []
    tensor_scores = []

    # search the best answer
    #     for query, query_embedding in zip(queries, query_embeddings):
    cosine_scores = util.pytorch_cos_sim(query_embeddings, question_list_emb)[0]
    results = zip(range(len(cosine_scores)), cosine_scores)
    # 第一个是按照score排序的index，第二个为对应的score但是是tensor格式
    results = sorted(results, key=lambda x: x[1], reverse=True)
    end_time = time.time()
    wasting_time = end_time - start_time
    for index, tensor_score in results:
        index_ranked.append(index)
        tensor_scores.append(tensor_score)

    if tensor_scores[0] > threshold_SBERT:
        if_valid = 1

    # 回答答案
    print('top few questions(TFIDF: %d) similar to "%s"' % (topk_SBERT, colored(query, 'green')))
    print("The best similarity for SBERT is:", tensor_scores[0])

    # 得到前几个的index
    topk_idx_SBERT = index_ranked[:topk_SBERT]

    for index, idx in enumerate(topk_idx_SBERT):
        print('SBERT; %s\t%s' % (colored('%.4f' % tensor_scores[index], 'cyan'), colored(question_list[idx], 'yellow')))

    if if_valid:
        print(answer_list[index_ranked[0]])

    return if_valid, topk_idx_SBERT, tensor_scores, wasting_time


def use_model_qa(model_path, QA_path):
    print("数据准备中")
    model = SentenceTransformer(model_path)

    topk_SBERT = 3
    threshold_SBERT = 0.6

    # data embedding
    question_list, answer_list = read_and_split_the_excel(QA_path)
    question_embeddings = model.encode(question_list, convert_to_tensor=True)
    print("准备完毕")
    init_cost_time = 0
    repeated_time = 100
    for i in range(repeated_time):
        # 获得问题
        query = "UIC处于哪个风水宝地职之位置呢?"
        # query = input("请输入问题（输入quit退出）:")
        if query == "quit":
            break

        if_valid, topk_idx_SBERT, tensor_scores, wasting_time = SBERT_get_reply(model, query, question_list, answer_list, question_embeddings, topk_SBERT, threshold_SBERT)

        init_cost_time += wasting_time

    mean_time = init_cost_time / repeated_time
    # print("Final mean time", init_cost_time / repeated_time)

    return mean_time
# # Test for all
# print("数据准备中")
# model = SentenceTransformer(model_path)

# topk_SBERT = 3
# threshold_SBERT = 0.7

# # data embedding
# question_list,answer_list = read_and_split_the_excel(QA_path)
# question_embeddings = model.encode(question_list,convert_to_tensor=True)
# print("准备完毕")
# # 获得问题
# q = "UIC是"
# SBERT_get_reply(model, q, question_list, answer_list, question_embeddings, topk_SBERT, threshold_SBERT)

def SBERT_QA():
    model_path_list = ['./SBert/sbert_base_chinese','./SBert/SBert_CN_fine_tune','./SBert/sn_xlm_roberta_base','./SBert/roberta_fine_tune','./SBert/deberta_v3_base_qa','./SBert/deberta_fine_tune']

    model_path = '.\SBert\sbert_base_chinese'
    # model_path = '.\SBert\SBert_CN_fine_tune'

    # model_path = '.\SBert\sn_xlm_roberta_base'
    # model_path = '.\SBert\roberta_fine_tune'

    # model_path = '.\SBert\deberta_v3_base_qa'
    # model_path = '.\SBert\deberta_fine_tune'

    QA_path = ".\dataset\CN_QA_dataset_all.xlsx"

    mean_time_list = []

    for model_name in model_path_list:
        mean_time = use_model_qa(model_name, QA_path)
        mean_time_list.append(mean_time)

    for mean_time_get in mean_time_list:
        print("The model has the mean time of", mean_time_get)

if __name__ == '__main__':
    SBERT_QA()
