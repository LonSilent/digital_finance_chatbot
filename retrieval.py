import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci
import jieba

jieba.set_dictionary('../zh-dict/dict_zh_small.txt')

jieba.load_userdict('../zh-dict/acg.txt')
jieba.load_userdict('../zh-dict/ec_item_zh.txt')
jieba.load_userdict('../zh-dict/user_dict_zh.txt')
jieba.load_userdict('./dict/finance_dict.txt')

def build_index(data_path):
    qa_path = 'qa.json'

    with open(qa_path) as f:
        data = json.load(f)

    question = []
    answer = []
    for d in data:
        question.append(' '.join(d[0]))
        answer.append(' '.join(d[1]))

    tv = TfidfVectorizer()
    tv.fit(question)

    features_vec = tv.transform(question)
    cp = ci.MultiClusterIndex(features_vec, answer)

    return tv, cp

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

    tv, cp = build_index('qa.json')

    search_data = [
        '我想要買基金，成本大概多少',
        '期貨要怎樣買？'
    ]

    search_data_cut = [' '.join(jieba.cut(x)) for x in search_data]
    search_features_vec = tv.transform(search_data_cut)

    for i in range(len(search_data)):
        print(search_data[i])
        print()
        result = cp.search(search_features_vec.getrow(i), k=1, return_distance=False)
        print(result[0][0].replace(' ',''))
        print("==================================")
    
