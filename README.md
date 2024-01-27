# UIC_QA_Model
The model and evaluation for the FYP

## File Description
### back_end_alg
The file mainly cosist of the Back end algorithm, and it needs the model obtain from [hugging face](https://huggingface.co/models) which consist of: 

1. bert-base-chinese
2. roberta-chinese-base
3. sn-xlm-roberta-base
4. sbert-base-chinese
5. deberta-v3-base-qa

Besides, it also contains the fine-tuning model which can download by the following link:

1. [SBERT after fine-tune](https://www.kaggle.com/code/shadowcattin/ealuation-sbert-finet-fyp/data)
2. [SRoBERTa after fine-tune](https://www.kaggle.com/code/shadowcattin/ealuation-sbert-ro-fit-fyp/data)
3. [SDeBERTa after fine-tune](https://www.kaggle.com/code/shadowcattin/ealuation-sbert-de-finet-fyp/data)

### fine-tune
This folder contain the our fine-tuning for the SBERT. We have fine-tuned the following three SBERT models, and the corresponding source files can be found on kaggle.

1. [sbert-base-chinese](https://huggingface.co/uer/sbert-base-chinese-nli)
2. [sn-xlm-roberta-base](https://huggingface.co/symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli)
3. [deberta-v3-base-qa](https://huggingface.co/jamescalam/deberta-v3-base-qa)

The fine-tuning kaggle file can be found by the following hyperlink:

1. [sbert-base-chinese](https://www.kaggle.com/shadowcattin/sentence-embedding-fyp)
2. [sn-xlm-roberta-base](https://www.kaggle.com/shadowcattin/sentence-embedding-roberta-fyp)
3. [deberta-v3-base-qa](https://www.kaggle.com/shadowcattin/sentence-embedding-deberta-fyp)

### evaluation
This folder mainly talk about how we evaluate the model. We totally evaluate the performance of the QA system by using the method of TF-IDF, BERT, SBERT separatly. 
The jupyter notebook are store in the evaluation folder, which includes 8 models. The source file can be found in the kaggle by using the hyperlink below.

1. [TF-IDF](https://www.kaggle.com/code/shadowcattin/ealuation-tfidf-fyp)
2. [BERT](https://www.kaggle.com/code/shadowcattin/ealuation-bert-emb-fyp)
3. [RoBERTa](https://www.kaggle.com/code/shadowcattin/sentence-embedding-roberta-fyp)
4. [SBERT](https://www.kaggle.com/code/shadowcattin/ealuation-sbert-base-fyp)
5. [SRoBERTa](https://www.kaggle.com/code/shadowcattin/ealuation-sbert-ro-fit-fyp)
6. [SDeBERTa](https://www.kaggle.com/code/shadowcattin/ealuation-sbert-deber-fyp)
7. [SBERT after fine-tune](https://www.kaggle.com/code/shadowcattin/ealuation-sbert-finet-fyp)
8. [SRoBERTa after fine-tune](https://www.kaggle.com/code/shadowcattin/ealuation-sbert-ro-fit-fyp)
9. [SDeBERTa after fine-tune](https://www.kaggle.com/code/shadowcattin/ealuation-sbert-de-finet-fyp)



