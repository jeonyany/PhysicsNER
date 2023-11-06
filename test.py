import predict


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


if __name__ == '__main__':
    text = input()
    filepath = './data/my_test.txt'
    write_input_data(text, filepath)
    predict.main()
    get_entity(filepath)
