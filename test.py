import os

import subprocess

input_dir = './data'
output_dir = './output'

chinese_entity_dic = {'OBJ': '物体', 'MOV': '场景', 'PRO': '属性', 'NUM': '数值'}

def write_input_data(text, file_path):
    text_list = list(text)
    output_predict_file = open(file_path, 'w', encoding='utf-8')
    for i in range(len(text_list)):
        s = text_list[i]
        output_predict_file.write(s + ' O' + '\n')
    output_predict_file.close()


def get_entity(file_path):
    f = file_path
    with open(f, 'r', encoding='utf-8') as f:
        sent, tag_list = [], []
        for line in f.readlines():
            sent.append(line.split()[0])
            tag_list.append(line.split()[1])
        print(sent)  # 句子分字列表
        print(tag_list)  # 标签列表
    entity_dict = {'OBJ': [], 'MOV': [], 'PRO': [], 'NUM': []}
    i = 0
    for char, tag in zip(sent, tag_list):
        # 标签匹配计算
        if 'B-' in tag:
            entity = char
            j = i + 1
            entity_type = tag.split('-')[-1]
            while j < min(len(sent), len(tag_list)) and 'I-%s' % entity_type in tag_list[j]:
                entity += sent[j]
                j += 1
            entity_dict[entity_type].append(entity)
        i += 1
    return entity_dict

def write_analysis_result(entity_dict):
    f = open(os.path.join(output_dir, "my_res.txt"), 'w', encoding='utf-8')
    count = 0
    for key in entity_dict:
        count += len(entity_dict[key])
    f.write(str(count))
    f.write('\n')
    print(count)
    for key in entity_dict:
        for i in range(len(entity_dict[key])):
            f.write(chinese_entity_dic[key] + '\t' + entity_dict[key][i] + '\n')
            print(chinese_entity_dic[key] + '\t' + entity_dict[key][i])
    f.close()

if __name__ == '__main__':
    text = input()
    filename = 'my_test.txt'
    write_input_data(text, os.path.join(input_dir, filename))
    subprocess.run(["python", "./predict.py"])
    entity_dic = get_entity(os.path.join(output_dir, filename))
    print(entity_dic)
    write_analysis_result(entity_dic)
