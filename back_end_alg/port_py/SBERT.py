# 安装包
# pip install -U sentence-transformers
# pip install -U transformers
# pip install openpyxl
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from copy import deepcopy
from random import randint
from termcolor import colored


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

    for index, tensor_score in results:
        index_ranked.append(index)
        tensor_scores.append(tensor_score)

    if tensor_scores[0] > threshold_SBERT:
        if_valid = 1

    # 回答答案
    # print('top few questions(TFIDF: %d) similar to "%s"' % (topk_SBERT, colored(query, 'green')))
    # print("The best similarity for TF-IDF is:", tensor_scores[0])

    # 得到前几个的index
    topk_idx_SBERT = index_ranked[:topk_SBERT]
    # return question_list[topk_idx_SBERT[0]]

    # 返回对应的答案
    return answer_list[index_ranked[0]]

    # for index, idx in enumerate(topk_idx_SBERT):
    #     print('SBERT; %s\t%s' % (colored('%.4f' % tensor_scores[index], 'cyan'), colored(question_list[idx], 'yellow')))
    #
    # if if_valid:
    #     print(answer_list[index_ranked[0]])
    #

    # return if_valid, topk_idx_SBERT, tensor_scores


def use_model_qa(model_path, QA_path):
    print("数据准备中")
    model = SentenceTransformer(model_path)

    topk_SBERT = 3
    threshold_SBERT = 0.6

    # data embedding
    question_list, answer_list = read_and_split_the_excel(QA_path)
    question_embeddings = model.encode(question_list, convert_to_tensor=True)
    print("准备完毕")
    while (1):
        # 获得问题
        query = input("请输入问题（输入quit退出）:")
        if query == "quit":
            break

        SBERT_get_reply(model, query, question_list, answer_list, question_embeddings, topk_SBERT, threshold_SBERT)


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
    model_path = '.\sbert_base_chinese'
    QA_path = ".\dataset\CN_QA_dataset_all.xlsx"
    use_model_qa(model_path, QA_path)


if __name__ == '__main__':
    SBERT_QA()
