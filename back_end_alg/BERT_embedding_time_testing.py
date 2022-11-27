import pandas as pd
import numpy as np
from termcolor import colored
from transformers import BertTokenizer, BertModel
import warnings
import time
warnings.filterwarnings("ignore")

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

def transfer_sentence_vector(sentence,tokenizer,model):
    """
    :func: 把句子embedding成向量
    :param sentence: 句子
    :param tokenizer: 分词器
    :param model: 模型
    :return: 转成的向量
    """
    # generate question vector
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)[1].detach().numpy()
    #第零个表示的是整个句子的信息
    return output.tolist()[0]

# # 测试
# tokenizer = BertTokenizer.from_pretrained('./bert_base_chinese')
# model = BertModel.from_pretrained('./bert_base_chinese')
# sentence = '学校现在有多少在校生？'
# print(transfer_sentence_vector(sentence,tokenizer,model))

def transfer_all_q2v(sentence_list,tokenizer,model):
    """
    :func: 把句子list都embedding成向量
    :param sentence: 句子的list
    :param tokenizer: 分词器
    :param model: 模型
    :return: 转成的向量list
    """
    doc_vecs=[]
    for sentence in sentence_list:
        doc_vecs.append(transfer_sentence_vector(sentence,tokenizer,model))
    doc_vecs = np.array(doc_vecs)
    return doc_vecs

# # 测试
# tokenizer = BertTokenizer.from_pretrained('./bert_base_chinese')
# model = BertModel.from_pretrained('./bert_base_chinese')
# question_list,answer_list = read_and_split_the_excel("./dataset/CN_QA_dataset_all.xlsx")
# doc_vecs = transfer_all_q2v(question_list,tokenizer,model)
# print(doc_vecs)


def get_similar_q_id(query_vec, doc_vecs, tokenizer, model, topk=5, threshold=0.95, all_score_without_rank=0):
    """
    :func: 通过cosine similarity找到相似句子
    :param sentence: 转为向量的句子
    :param doc_vecs: 已经转换为向量的句子列表
    :param topk: 显示前topk个最相似的句子
    :param threshold: 认为是匹配的问句的有效阈值
    :param tokenizer: 分词器
    :param model: 模型
    :return: 是否达到要求，返回满足阈值要求的问题所在行索引——对应答案所在的行索引的np.array，相似度分数
    """
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1) / np.linalg.norm(query_vec)

    if all_score_without_rank:
        return score
    else:
        # get the top "topk" score's id
        topk_idx = np.argsort(score)[::-1][:topk]

        # if the score is larger than the threshold
        if score[topk_idx[0]] < threshold:
            if_vaild = 0
        else:
            if_vaild = 1

        return if_vaild, topk_idx, score

# 测试
# tokenizer = BertTokenizer.from_pretrained('./bert_base_chinese')
# model = BertModel.from_pretrained('./bert_base_chinese')
# question_list,answer_list = read_and_split_the_excel("./dataset/CN_QA_dataset_all.xlsx")
# sentence = '学校现在有多少在校生？'
# sentence_vec = transfer_sentence_vector(sentence,tokenizer,model)
# topk = 5
# threshold = 0.95
#
# doc_vecs = transfer_all_q2v(question_list,tokenizer,model)
# if_vaild, topk_idx, score = get_similar_q_id(sentence_vec,doc_vecs,tokenizer,model,topk,threshold)
# print(if_vaild)
# print(topk_idx)
# print(score[:5])

def Bert_em_prepared(data_path, model_path):
    # initial things
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    question_list, answer_list = read_and_split_the_excel(data_path)
    doc_vecs = transfer_all_q2v(question_list, tokenizer, model)

    return tokenizer, model, question_list,answer_list, doc_vecs

def Bert_em_reply(query,tokenizer, model, question_list,answer_list, doc_vecs, topk=5, threshold=0.95):
    start_time = time.time()
    query_vec = transfer_sentence_vector(query, tokenizer, model)

    # 匹配
    if_vaild, Bert_emb_topk_idx, Bert_emb_each_score = get_similar_q_id(query_vec, doc_vecs, tokenizer, model, topk, threshold)

    end_time = time.time()

    wasting_time = end_time - start_time
    # 返回最相似的问题
    print('top %d questions similar to "%s"' % (topk, colored(query, 'green')))
    for idx in Bert_emb_topk_idx:
        print('&gt; %s\t%s' % (colored('%.4f' % Bert_emb_each_score[idx], 'cyan'), colored(question_list[idx], 'yellow')))
    print("The best similarity is:", Bert_emb_each_score[Bert_emb_topk_idx[0]])

    # get the answer
    if if_vaild:
        print(answer_list[Bert_emb_topk_idx[0]])
    return if_vaild, Bert_emb_topk_idx, Bert_emb_each_score,wasting_time

def Bert_em_QA():
    # preparing
    print("数据准备中")

    data_path = "./dataset/CN_QA_dataset_all.xlsx"

    # model_path = './emb_bert/bert_base_chinese'
    model_path = './emb_bert/roberta_chinese_base'

    tokenizer, model, question_list, answer_list, doc_vecs = Bert_em_prepared(data_path, model_path)

    Bert_emb_topk = 5
    Bert_emb_threshold = 0.95

    print("准备完毕")
    # get the query
    init_cost_time = 0
    repeated_time = 100
    for i in range(repeated_time):
        # 读取数据
        query = "UIC处于哪个风水宝地职之位置呢?"
        if query == "quit":
            break

        # get the reply
        if_vaild, Bert_emb_topk_idx, Bert_emb_each_score,wasting_time = Bert_em_reply(query, tokenizer, model, question_list, answer_list, doc_vecs, Bert_emb_topk, Bert_emb_threshold)

        init_cost_time += wasting_time

    print("Final mean time", init_cost_time / repeated_time)

if __name__ == '__main__':
    Bert_em_QA()