import json

# 五言绝句 num=20  七言绝句 七言律句 num=50
def preprocess_file(raw_data_path, num):
    # 语料文本内容
    files_content = ''
    with open(raw_data_path, 'r', encoding='UTF-8') as f:
        lines = json.load(f)
        for line in lines:
            x = line.strip() + "]"
            files_content += x

    words = sorted(list(files_content))
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] < num:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])

    words, _ = zip(*wordPairs)
    # words += (" ",)
    words = list(words)
    words.pop(words.index(']'))
    # print(words)
    # print(len(words))

    # word到id的映射
    word2num = dict((c, i) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    word2numF = lambda x: word2num.get(x, len(words) - 1)
    return word2numF, num2word, words, files_content