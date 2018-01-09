import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci
import jieba
from collections import defaultdict

jieba.set_dictionary('./dict/dict_zh_small.txt')

jieba.load_userdict('./dict/user_dict_zh.txt')
jieba.load_userdict('./dict/finance_dict.txt')

def build_index(data_path):
    qa_path = 'qa.json'

    with open(qa_path) as f:
        data = json.load(f)

    question = []
    answer = []
    for d in data:
        question.append(' '.join(d[0]) + ' '.join(d[1]))
        answer.append(' '.join(d[1]))

    tv = TfidfVectorizer()
    tv.fit(question)

    features_vec = tv.transform(question)
    cp = ci.MultiClusterIndex(features_vec, answer)

    return tv, cp

def build_index_new(data_path):
    qa_path = 'qa_final.json'

    with open(qa_path) as f:
        data = json.load(f)

    question = defaultdict(list)
    answer = defaultdict(list)
    all_question = []
    all_cp = {}
    for k, v in data.items():
        for pair in v:
            question[k].append(' '.join(pair[0]))
            # question[k].append(' '.join(pair[0]) + ' '.join(pair[1]))
            answer[k].append(' '.join(pair[1]))
            # all_question.append(' '.join(pair[0]) + ' '.join(pair[1]))
            all_question.append(' '.join(pair[0]))


    # print(question['信用卡'][0])
    # exit()

    tv = TfidfVectorizer()
    tv.fit(all_question)

    for k, v in question.items():
        features_vec = tv.transform(v)
        cp = ci.MultiClusterIndex(features_vec, answer[k])
        all_cp[k] = cp

    return tv, all_cp

if __name__ == '__main__':
    # qa_path = 'qa.json'

    # with open(qa_path) as f:
    #     data = json.load(f)

    # question = []
    # answer = []
    # for d in data:
    #     question.append(' '.join(d[0]))
    #     answer.append(' '.join(d[1]))

    # tv = TfidfVectorizer()
    # tv.fit(question)

    # features_vec = tv.transform(question)
    # cp = ci.MultiClusterIndex(features_vec, answer)

    tv, all_cp = build_index_new('qa_final.json')
    # tv, cp = build_index('qa.json')

    search_data = [
        '信用卡盜刷'
    ]

    search_data_cut = [' '.join(jieba.cut(x)) for x in search_data]
    search_features_vec = tv.transform(search_data_cut)

    # for i in range(len(search_data)):
    #     print(search_data[i])
    #     print()
    #     result = cp.search(search_features_vec.getrow(i), k=1, return_distance=False)
    #     print(result[0][0].replace(' ',''))
    #     print("==================================")
    
    for i in range(len(search_data)):
        print(search_data[i])
        print()
        result = all_cp['外匯'].search(search_features_vec.getrow(i), k=1, return_distance=False)
        print(result[0][0].replace(' ',''))
        print("==================================")

