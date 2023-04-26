import random
import os

import keras
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model
from keras.layers import LSTM, Dropout, Dense, Input
from keras.optimizers import Adam
from main.utils.rhyme_table.get_rhyme import get_word_rhyme, get_word_foots, flat_oblique_tone_table
from main.utils.rhyme_table.get_rhyme import pingshui2word, word2pingshui, xinyun2word, word2xinyun
from main.utils.model.lstm.utils import preprocess_file
from random import shuffle

def shuffle_str(s): # 打乱字符串
    # 将字符串转换成列表
    str_list = list(s)
    # 调用random模块的shuffle函数打乱列表
    shuffle(str_list)
    # 将列表转字符串
    return ''.join(str_list)


class LstmModel(object):
    def __init__(self, yan, jue, ru, qi, use_rhyme, model_path, raw_data_path):
        self.use_rhyme = use_rhyme
        self.yan = yan # 7:七言 5:五言
        self.raw_data_path = raw_data_path
        self.model_path = model_path
        self.max_len = self.yan + 1
        self.jue = jue  # 0:绝句 1:律
        self.ru = ru  # 0:首句入韵 1:首句不入
        self.qi = qi  # 0:仄起 1:平起
        self.tune = flat_oblique_tone_table[0 if self.yan == 7 else 1][self.jue][self.ru][self.qi]  # 诗韵表

        self.batch_size = 32
        self.learning_rate = 0.001
        self.model = None
        self.do_train = True
        self.loaded_model = True
        self.temperature = 1 # temperature越小 字之间关联性越大 但是作诗耗费时间越长

        if self.yan == 5 and self.jue == 0:
            num = 20
        else:
            num = 50
        # 文件预处理
        self.word2numF, self.num2word, self.words, self.files_content = preprocess_file(self.raw_data_path, num)
        # 诗的list
        self.poems = self.files_content.split(']')
        # 诗的总数量
        self.poems_num = len(self.poems)

        # 如果模型文件存在则直接加载模型
        if os.path.exists(self.model_path) and self.loaded_model:
            self.model = load_model(self.model_path)

        # 压哪个韵 0:不压 1:平水 2:新韵
        self.use_rhyme = self.use_rhyme
        if self.use_rhyme == 1:
            self.word2rhyme = word2pingshui
            self.rhyme2word = pingshui2word['平声']
        elif self.use_rhyme == 2:
            self.word2rhyme = word2xinyun
            self.rhyme2word = xinyun2word['平声']

        # 七言 6, 14, 30, 46, 62
        # 五言 4, 10, 22, 34, 46
        self.foot = ''  # 韵脚
        self.idx = 0
        self.last_word = ''  # 上一个字，防止相同
        self.jiao = set()  # 韵脚不能重字
        self.word_dict = {}  # 防止生成的诗句太多重复字

        print('__int__')

    '''初始化'''
    def init_step(self, text, idx):
        self.idx = idx
        self.jiao = set()
        self.word_dict = {}
        for word in text:  # 将text中的字加入诗词的用字情况
            if word in self.word_dict:
                self.word_dict[word] += 1
            else:
                self.word_dict[word] = 1

    '''按照押韵条件过滤'''
    def rhyme_filter(self, word):

        # 无论如何不能用已用过的韵脚字
        if word in self.jiao:
            return False

        max_len = self.max_len

        # 韵脚处理

        # 1.首句入韵，处理首韵-第一句 2.首句不入韵，处理首韵-第二句
        if (self.idx == max_len - 2 and self.ru == 0) or (self.idx == 2 * max_len - 2 and self.ru == 1):
            if len(get_word_foots(word, self.word2rhyme)):
                self.foot = get_word_foots(word, self.word2rhyme)[0]
                print("韵脚：", self.foot)
                self.jiao.add(word)
                return True
            else:
                return False

        # 1.不管哪种情况，第四句一定是会有，且之前已经确定好韵脚 2.如果是首句入韵的情况，第二句需要向第一句的韵脚看齐
        if self.idx == 4 * max_len - 2 or (self.idx == 2 * max_len - 2 and self.ru == 0):
            foots = get_word_foots(word, self.word2rhyme)
            if self.foot not in foots:  # 压同韵脚
                return False
            else:
                self.jiao.add(word)
                return True

        # 如果是律，还有两联需要押韵脚
        if self.idx == 6 * max_len - 2 or self.idx == 8 * max_len - 2:
            foots = get_word_foots(word, self.word2rhyme)
            if self.foot not in foots or word in self.jiao:
                return False
            else:
                self.jiao.add(word)
                return True

        # 非韵脚处理
        ping, ze = get_word_rhyme(word, self.word2rhyme)

        if self.foot != '' and self.foot in ping: # 非韵脚处不能占用韵脚调的字
            return False

        if self.tune[self.idx] == 'x':
            pass
        elif self.tune[self.idx] == '0' and len(ping) > 0:
            pass
        elif self.tune[self.idx] == '1' and len(ze) > 0:
            pass
        elif self.tune[self.idx] == '/':
            pass
        else:
            return False

        return True

    '''按照叠词，字使用次数过多 等条件过滤'''
    def base_filter(self, word):
        if self.tune[self.idx] == '/': # 标点符号直接跳过
            return True
        # 唐诗七言：黯  唐诗五言：魔  宋诗五言：馥  宋诗七言：隆  "不"音太多
        # if word == "黯" or word == "魔" or word == "馥" or word == "隆" or word == "不":
        #     return False

        # 七言绝句：鹜  七言律句：𦶟  五言绝句：鶑  五言律句：𫛳
        if word == "鹜" or word == "𦶟" or word == "鶑" or word == "𫛳" or word == "不":
            return False

        # 如果这个字已经出现过不止一次，则不能用
        if word in self.word_dict and self.word_dict[word] > 0:
            return False

        # 不和上一个字重复，不叠词
        if word == self.last_word:
            return False

        return True

    '''建立模型'''
    def build_model(self):
        print('building model_file')

        # 输入的dimension
        input_tensor = Input(shape=(self.max_len, len(self.words)))
        lstm = LSTM(512, return_sequences=True)(input_tensor)
        # lstm = CuDNNLSTM(512, return_sequences=True)(input_tensor)
        dropout = Dropout(0.6)(lstm)
        lstm = LSTM(256)(dropout)
        # lstm = CuDNNLSTM(256)(dropout)
        dropout = Dropout(0.6)(lstm)
        dense = Dense(len(self.words), activation='softmax')(dropout) # Dense：全连接层 softmax：归一化（所有维度概率合为1）
        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    '''
        当temperature=1.0时，模型输出正常
        当temperature=1.5时，模型输出比较open
        当temperature=0.5时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        就是一个概率分布变换的问题，保守的时候概率大的值变得更大，选择的可能性也更大
    '''
    def sample(self, preds):

        preds = np.asarray(preds).astype('float64')
        exp_preds = np.power(preds, 1. / self.temperature)
        preds = exp_preds / np.sum(exp_preds)
        pro = np.random.choice(range(len(preds)), 1, replace=False, p=preds) # 不放回取样

        num = int(pro.squeeze())
        word = self.num2word[num]
        count = 0

        if self.use_rhyme: # 用韵生成格律诗
            while not (self.base_filter(word) and self.rhyme_filter(word)) and count < 10000:  # 若生成的字符不满足押韵和基础要求 则重新生成
                pro = np.random.choice(range(len(preds)), 1, replace=False, p=preds) # 不放回取样
                num = int(pro.squeeze())
                word = self.num2word[num]
                count += 1
            pass
        else: # 不用韵
            while not (self.base_filter(word)) and count < 10000:  # 若生成的字符不满足押韵和基础要求 则重新生成
                pro = np.random.choice(range(len(preds)), 1, replace=False, p=preds) # 不放回取样
                num = int(pro.squeeze())
                word = self.num2word[num]
                count += 1


        if count >= 1000:
            print(f"{self.foot}:{count},{word}")
            return -1

        if word != '，' and word != '。':  # 标点符号不算
            if word in self.word_dict:  # 记录诗句的用字情况
                self.word_dict[word] += 1
            else:
                self.word_dict[word] = 1

            self.last_word = word  # 更新last_word

        self.idx += 1  # 预测下一个字符

        return num

    '''内部使用方法，根据一串输入，返回单个预测字符'''
    def _pred(self, sentence):
        max_len = self.max_len
        if len(sentence) < max_len:
            print('in def _pred,length error ')
            return

        x_pred = np.zeros((1, max_len, len(self.words)))
        for t, word in enumerate(sentence):
            x_pred[0, t, self.word2numF(word)] = 1.
        preds = self.model.predict(x_pred, verbose=0)[0]

        next_index = self.sample(preds)
        next_word = ' '

        if next_index == -1: #  如果实在找不到韵脚
            count = 0
            ping_dict = [word[0] for word in self.rhyme2word[self.foot][0]]
            str = shuffle_str(ping_dict)  # 打乱一下字符 免得每次按顺序找到的是同一个字
            for word in str: # 直接跳过模型 从韵表里面取
                count += 1
                if self.base_filter(word) and self.rhyme_filter(word) and word in self.words: # 这个韵脚不能是生僻字 最好在训练数据里面有
                    next_word = word
                    self.idx += 1 # 找到了 idx++
                    break
            print(f"韵脚找了：{count}次，{next_word}")
            pass
        else:
            next_word = self.num2word[next_index]

        return next_word

    '''
        sentence:预测输入值
        lenth:预测出的字符串长度
        供类内部调用，输入max_len长度字符串，返回length长度的预测值字符串
    '''
    def _preds(self, sentence, length):

        sentence = sentence[:self.max_len]
        generate = ''
        for i in range(length):
            pred = self._pred(sentence)  # 根据前八个字符生成下一个字符
            generate += pred
            sentence = sentence[1:] + pred  # 窗口滑动
            # print(sentence)
        return generate

    '''训练过程中，每4个epoch打印出当前的学习情况'''
    def generate_sample_result(self, epoch, logs):

        if epoch % 4 != 0:
            return

        with open('out/out.txt', 'a', encoding='utf-8') as f:
            f.write('==================Epoch {}=====================\n'.format(epoch))

        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.7, 1.0, 1.3]:
            print("------------Diversity {}--------------".format(diversity))
            self.temperature = diversity
            generate = self.predict_random()
            print(generate)

            # 训练时的预测结果写入txt
            with open('out/out.txt', 'a', encoding='utf-8') as f:
                f.write(generate + '\n')

    '''随机从库中选取一句开头的诗句，生成七言绝句'''
    def predict_random(self):

        if not self.model:
            print('model_file not loaded')
            return

        index = random.randint(0, self.poems_num)
        sentence = self.poems[index][: self.max_len]
        generate = self.predict_sen(sentence)
        return generate

    '''根据给出的前max_len个字，生成诗句'''
    def predict_sen(self, text):
        '''此例中，即根据给出的第一句诗句（含逗号），来生成古诗'''
        if not self.model:
            return
        max_len = self.max_len
        if len(text) < max_len:
            print('length should not be less than ', max_len)
            return

        self.init_step(text, max_len)
        self.last_word = text[-2]

        if self.ru == 0 and self.use_rhyme != 0:
            self.foot = get_word_foots(text[-2], self.word2rhyme)[0]
            print("韵脚：", self.foot)
            self.jiao.add(text[-2])

        generate = text
        generate += self._preds(text, length=(1 + self.jue) * 4 * max_len - max_len)

        return self.use_rhyme, self.foot, generate

    '''根据给出的首个文字，生成七言绝句'''
    def predict_first(self, word):
        if not self.model:
            print('model_file not loaded')
            return

        self.init_step(word, 1)
        self.last_word = word

        index = random.randint(0, self.poems_num)
        # 选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
        sentence = self.poems[index][1 - self.max_len:] + word
        generate = str(word)

        # 直接预测后面字符
        generate += self._preds(sentence, length=(self.jue + 1) * 4 * self.max_len - 1)
        return self.use_rhyme, self.foot, generate

    '''根据给4个字，生成藏头诗七言绝句'''
    def predict_hide(self, text):
        if not self.model:
            print('model_file not loaded')
            return
        if len(text) != 4:
            print('藏头诗的输入必须是4个字！')
            return

        self.jue = 0  # 藏头诗只生成绝句
        max_len = self.max_len
        self.init_step(text, 1)
        self.last_word = text[0]

        index = random.randint(0, self.poems_num)
        # 选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
        sentence = self.poems[index][1 - self.max_len:] + text[0]
        generate = str(text[0])

        for i in range(max_len - 1):
            next_word = self._pred(sentence)
            sentence = sentence[1:] + next_word
            generate += next_word

        for i in range(3):
            generate += text[i + 1]
            sentence = sentence[1:] + text[i + 1]
            self.idx += 1

            for i in range(max_len - 1):
                next_word = self._pred(sentence)
                sentence = sentence[1:] + next_word
                generate += next_word

        return self.use_rhyme, self.foot, generate

    '''生成器生成数据'''
    def data_generator(self):
        max_len = self.max_len
        i = 0
        while 1:
            x = self.files_content[i: i + max_len]
            y = self.files_content[i + max_len]

            if ']' in x or ']' in y:
                i += 1
                continue

            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool_
            )
            y_vec[0, self.word2numF(y)] = 1.0

            x_vec = np.zeros(
                shape=(1, self.max_len, len(self.words)),
                dtype=np.bool_
            )

            for t, word in enumerate(x):
                x_vec[0, t, self.word2numF(word)] = 1.0

            yield x_vec, y_vec
            i += 1

    '''训练模型'''
    def train(self):
        print('training')
        number_of_epoch = len(self.files_content) - (self.max_len + 1) * self.poems_num
        number_of_epoch /= self.batch_size
        number_of_epoch = int(number_of_epoch / 1.5)
        print('epoches = ', number_of_epoch)
        print('poems_num = ', self.poems_num)
        print('len(self.files_content) = ', len(self.files_content))

        if not self.model:
            self.build_model()

        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.batch_size,
            epochs=number_of_epoch,
            # callbacks=None,
            # callbacks=[
            #     keras.callbacks.ModelCheckpoint(self.model_path, save_weights_only=False, period=1),
            #     LambdaCallback(on_epoch_end=self.generate_sample_result)
            # ]

        )
        self.model.save(self.model_path)
        # self.model_file.save_weights(self.model_path)