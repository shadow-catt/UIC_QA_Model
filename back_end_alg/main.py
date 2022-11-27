from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from sentence_transformers import SentenceTransformer
from SBERT import read_and_split_the_excel, SBERT_get_reply
from BERT_embedding import Bert_em_prepared, Bert_em_reply
from tfidf import TF_IDF_prepared, TF_IDF_reply
from greeting import greet_start, greet_pre, greet_check, greet_none_reply
from translate import translate
import langid

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

print("数据准备中")
data_path = "./dataset/CN_QA_dataset_all.xlsx"
model_Bert_path = './emb_bert/roberta_base_chinese'
stopword_file_path = './dataset/cn_stopwords.txt'
model_SBert_path = './SBert/sbert_base_chinese'
model_SBert = SentenceTransformer(model_SBert_path)

topk_SBert = 3
threshold_SBert = 0.6
question_list_SBert, answer_list_SBert = read_and_split_the_excel(data_path)
question_embeddings_SBert = model_SBert.encode(question_list_SBert, convert_to_tensor=True)

tokenizer, model, question_list, answer_list, doc_vecs = Bert_em_prepared(data_path, model_Bert_path)
topk_Bert = 5
threshold_Bert = 0.95

topk_TFIDF = 3
threshold_TFIDF = 0.7
question_list_tfidf, question_token_list, answer_list_tfidf, stop_words = TF_IDF_prepared(data_path, stopword_file_path)

appid = '20220713001272202'  # 填写你的appid
secretKey = 'v4x3PC6W6X9etHyXO5wu'  # 填写你的密钥


@app.route("/")
@app.route("/query-sbert", methods=['POST', 'GET'])
def query_sbert():
    try:
        data = json.loads(request.data.decode(encoding='utf8'))
        query = data.get('query', None)
        if query:
            lang = langid.classify(query)[0]
            if lang in ['zh', 'en', 'de']:
                query = translate(query, lang, 'zh', appid, secretKey)
                result, if_valid = SBERT_get_reply(model_SBert, query, question_list_SBert, answer_list_SBert,
                                                   question_embeddings_SBert, topk_SBert, threshold_SBert)
                result = translate(result, 'zh', lang, appid, secretKey)
                if if_valid:
                    greet = translate(greet_pre(), 'zh', lang, appid, secretKey)
                    return jsonify({"answer": greet + "\n" + result})
                else:
                    greet_none = translate(greet_none_reply(), 'zh', lang, appid, secretKey)
                    return jsonify({"answer": greet_none})
            else:
                return jsonify({"answer": "Sorry, this language is not supported yet!"})
        else:
            return jsonify({'answer': False, 'msg': '请输入您的问题哦！'}), 400
    except ValueError as ve:
        return jsonify({'status': False, 'msg': '输入参数格式不正确！'}), 400


@app.route("/query-bert", methods=['POST', 'GET'])
def query_bert():
    try:
        data = json.loads(request.data.decode(encoding='utf8'))
        query = data.get('query', None)
        if query:
            query = translate(query, 'auto', 'zh', appid, secretKey)
            result, if_valid = Bert_em_reply(query, tokenizer, model, question_list, answer_list, doc_vecs, topk_Bert, threshold_Bert)
            if if_valid:
                return jsonify({"answer": greet_pre() + "\n" + result})
            else:
                return jsonify({"answer": greet_none_reply()})
        else:
            return jsonify({'status': False, 'msg': '请输入您的问题哦！'}), 400
    except ValueError as ve:
        return jsonify({'status': False, 'msg': '输入参格式不正确！'}), 400


@app.route("/query-tfidf", methods=['POST', 'GET'])
def query_tfidf():
    try:
        data = json.loads(request.data.decode(encoding='utf8'))
        query = data.get('query', None)
        if query:
            query = translate(query, 'auto', 'zh', appid, secretKey)
            result, if_valid = TF_IDF_reply(query, question_list_tfidf, question_token_list, answer_list_tfidf, stop_words, topk_TFIDF, threshold_TFIDF)
            if if_valid:
                return jsonify({"answer": greet_pre() + "\n" + result})
            else:
                return jsonify({"answer": greet_none_reply()})
        else:
            return jsonify({'status': False, 'msg': '请输入您的问题哦！'}), 400
    except ValueError as ve:
        return jsonify({'status': False, 'msg': '输入参格式不正确！'}), 400


if __name__ == "__main__":
    app.run()