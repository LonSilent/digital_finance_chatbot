import glob
from openpyxl import load_workbook
import json
import jieba
from collections import defaultdict

jieba.set_dictionary('./dict/dict_zh_small.txt')

jieba.load_userdict('./dict/user_dict_zh.txt')
jieba.load_userdict('./dict/finance_dict.txt')

datapath = glob.glob('./data/database.xlsx')
result_path = 'qa_final.json'
content = defaultdict(list)

for path in datapath:
    wb = load_workbook(filename=path)
    sheets = wb.get_sheet_names()
    for sheet in sheets:
        work_sheet = wb.get_sheet_by_name(sheet)
        rows = work_sheet.rows
        columns = work_sheet.columns

        for row in rows:
            line = [col.value for col in row]
            if line[1] == None:
                pass
            else:
                line = [jieba.lcut(x) for x in line]
                content[sheet].append(line)

with open(result_path, 'w') as f:
    json.dump(content, f)

# sentence = []
# for qa in content:
#     question = [x for x in qa[0] if x != '\n' and x != ' ']
#     answer = [x for x in qa[1] if x != '\n' and x != ' ']

#     sentence.append(' '.join(question))
#     sentence.append(' '.join(answer))

# with open('sentence.txt', 'w') as f:
#     for line in sentence:
#         print(line, file=f)
