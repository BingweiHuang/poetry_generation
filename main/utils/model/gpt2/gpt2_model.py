import transformers
import torch
import torch.nn.functional as F
import os
import json
import random
from main.utils.model.gpt2 import tokenization_bert
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from tqdm import trange
from transformers import GPT2LMHeadModel
from poetry_generation.settings import BASE_DIR
from main.utils.rhyme_table.get_rhyme import flat_oblique_tone_table, get_word_rhyme, get_word_foots
from main.utils.rhyme_table.get_rhyme import pingshui2word, word2pingshui, xinyun2word, word2xinyun


def shuffle_str(s):  # 打乱字符串
    # 将字符串转换成列表
    str_list = list(s)
    # 调用random模块的shuffle函数打乱列表
    random.shuffle(str_list)
    # 将列表转字符串
    return ''.join(str_list)


class Gpt2Model(object):
    def __init__(self, yan, jue, ru, qi, use_rhyme, model_path, raw_data_path):
        self.device = '0,1,2,3'
        self.model_config = f'{BASE_DIR}/main/utils/model/gpt2/model_file/model_config_shi.json'  # 模型参数文件的路径
        self.tokenizer_path = f'{BASE_DIR}/main/utils/model/gpt2/model_file/vocab_shi.txt'  # 字典文件的路径
        self.tokenizer = tokenization_bert.BertTokenizer(vocab_file=self.tokenizer_path)  # 加载token的字典
        self.raw_data_path = raw_data_path  # 原始训练语料的路径
        self.tokenized_data_path = 'data/tokenized/'  # tokenized语料存放位置
        self.raw = True  # 是否先做tokenize
        self.epochs = 5  # 训练循环
        self.batch_size = 1  # 训练batch size
        self.lr = 1.5e-4  # 学习率
        self.warmup_steps = 2000  # warm up步数
        self.log_step = 10  # 多少步汇报一次loss
        self.stride = 768  # 训练时取训练数据的窗口步长
        self.gradient_accumulation = 1  # 梯度积累
        self.max_grad_norm = 1.0
        self.num_pieces = 100  # 将训练语料分成多少份
        self.min_length = 2  # 最短诗词长度
        self.output_dir = 'model/'  # 模型输出路径
        self.pretrained_model = ''  # 模型训练起点路径
        self.writer_dir = 'tensorboard_summary/'  # Tensorboard路径

        self.temperature = 1 # temperature越小 字之间关联性越大 但是作诗耗费时间越长
        self.topk = 200  # 最高几选一
        self.topp = 0  # 最高积累概率
        self.model_path = model_path  # 模型路径

        self.yan = yan  # 7:七言 5:五言
        self.jue = jue  # 0:绝句 1:律
        self.ru = ru  # 0:首句入韵 1:首句不入
        self.qi = qi  # 0:仄起 1:平起
        self.tune = flat_oblique_tone_table[0 if self.yan == 7 else 1][self.jue][self.ru][self.qi]  # 平仄调

        # 压哪个韵 0:不压 1:平水 2:新韵
        self.use_rhyme = use_rhyme
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

    '''训练数据分块'''

    def build_files(self, data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
        if not os.path.exists(tokenized_data_path):
            os.mkdir(tokenized_data_path)
        with open(data_path, 'r', encoding='utf8') as f:
            print('reading lines')
            lines = json.load(f)
            # lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
            all_len = len(lines)
        for i in tqdm(range(num_pieces)):
            sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
            if i == num_pieces - 1:
                sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
            sublines = [full_tokenizer.tokenize(line) for line in sublines if
                        len(line) >= min_length]  # 只考虑长度不少于min_length的句子
            sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
            full_line = []
            for subline in sublines:
                # full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
                full_line.extend(subline)
                # full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
                for id in full_line:
                    f.write(str(id) + ' ')
        print('finish')

    '''训练'''

    def train(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device  # 此处设置程序使用哪些显卡
        model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(self.model_config)
        n_ctx = model_config.n_ctx
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=self.tokenizer_path)
        full_tokenizer.max_len = n_ctx
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('using device:', device)

        raw_data_path = self.raw_data_path
        tokenized_data_path = self.tokenized_data_path
        raw = self.raw  # 选择是否从零开始构建数据集
        epochs = self.epochs
        batch_size = self.batch_size
        lr = self.lr
        warmup_steps = self.warmup_steps
        log_step = self.log_step
        stride = self.stride
        gradient_accumulation = self.gradient_accumulation
        max_grad_norm = self.max_grad_norm
        num_pieces = self.num_pieces
        min_length = self.min_length
        output_dir = self.output_dir
        tb_writer = SummaryWriter(log_dir=self.writer_dir)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if raw:
            print('building files')
            self.build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
                             full_tokenizer=full_tokenizer, min_length=min_length)
            print('files built')

        if not self.pretrained_model:
            model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
        else:
            model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(self.pretrained_model)
        model.to(device)

        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        print('number of parameters: {}'.format(num_parameters))

        multi_gpu = False
        full_len = 0
        print('calculating total steps')
        for i in tqdm(range(num_pieces)):
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                full_len += len([int(item) for item in f.read().strip().split()])
        total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)
        print('total steps = {}'.format(total_steps))

        optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
        scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                      t_total=total_steps)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = DataParallel(model)
            multi_gpu = True
        print('starting training')
        overall_step = 0
        for epoch in range(epochs):
            print('epoch {}'.format(epoch + 1))
            now = datetime.now()
            print('time: {}'.format(now))
            x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
            random.shuffle(x)
            piece_num = 0
            for i in x:
                running_loss = 0
                with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                    line = f.read().strip()
                tokens = line.split()
                tokens = [int(token) for token in tokens]
                start_point = 0
                samples = []
                while start_point < len(tokens) - n_ctx:
                    samples.append(tokens[start_point: start_point + n_ctx])
                    start_point += stride
                start_point -= stride
                last = tokens[start_point + n_ctx:]
                last.extend([full_tokenizer.convert_tokens_to_ids(['[PAD]']) * (n_ctx - len(last))])
                random.shuffle(samples)
                for step in range(len(samples) // batch_size):  # drop last

                    #  prepare data
                    batch = samples[step * batch_size: (step + 1) * batch_size]
                    batch_labels = []
                    batch_inputs = []
                    for ids in batch:
                        int_ids_for_labels = [int(x) for x in ids]
                        int_ids_for_inputs = [int(x) for x in ids]
                        batch_labels.append(int_ids_for_labels)
                        batch_inputs.append(int_ids_for_inputs)
                    batch_labels = torch.tensor(batch_labels).long().to(device)
                    batch_inputs = torch.tensor(batch_inputs).long().to(device)

                    #  forward pass
                    outputs = model.forward(input_ids=batch_inputs, labels=batch_labels)
                    loss, logits = outputs[:2]

                    #  get loss
                    if multi_gpu:
                        loss = loss.mean()
                    if gradient_accumulation > 1:
                        loss = loss / gradient_accumulation

                    #  loss backward

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    #  optimizer step
                    if (step + 1) % gradient_accumulation == 0:
                        running_loss += loss.item()
                        scheduler.step()
                        optimizer.step()
                        optimizer.zero_grad()
                        overall_step += 1
                        if (overall_step + 1) % log_step == 0:
                            tb_writer.add_scalar('loss', loss.item(), overall_step)
                    if (overall_step + 1) % log_step == 0:
                        print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                            datetime.now().hour,
                            datetime.now().minute,
                            (step + 1) // gradient_accumulation,
                            piece_num,
                            epoch + 1,
                            running_loss * gradient_accumulation / log_step))
                        running_loss = 0
                piece_num += 1

            print('saving model for epoch {}'.format(epoch + 1))
            if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
            # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
            # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
            print('epoch {} finished'.format(epoch + 1))

            then = datetime.now()
            print('time: {}'.format(then))
            print('time for one epoch: {}'.format(then - now))

        print('training finished')
        if not os.path.exists(output_dir + 'final_model'):
            os.mkdir(output_dir + 'final_model')
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'final_model')

    '''判断字符是否是汉字'''

    def _is_chinese_char(self, char):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        cp = ord(char)
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    '''按照押韵条件过滤'''

    def rhyme_filter(self, word):

        # 无论如何不能用已用过的韵脚字
        if word in self.jiao:
            return False

        max_len = self.yan + 1

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

        if self.foot != '' and self.foot in ping:  # 非韵脚处不能占用韵脚调的字
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
        if self.tune[self.idx] == '/':  # 标点符号直接跳过
            return True

        if not self._is_chinese_char(word[0]):  # 不是汉字，不能用
            print(f'{word}不是汉字')
            return False

        # 如果这个字已经出现过不止一次，则不能用
        if word in self.word_dict and self.word_dict[word] > self.jue:
            return False

        # 不和上一个字重复，不叠词
        if word == self.last_word:
            return False

        return True

    '''单个字sample选用'''

    def sample(self, filtered_logits):
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)  # 从备选里面sample一个
        word = self.tokenizer.convert_ids_to_tokens(int(next_token))  # 按照字典id找到预测的下一个字

        count = 0
        if self.use_rhyme:  # 用韵生成格律诗
            while not (self.base_filter(word) and self.rhyme_filter(word)) and count < 10000:  # 若生成的字符不满足押韵和基础要求 则重新生成
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)  # 从备选里面sample一个
                word = self.tokenizer.convert_ids_to_tokens(int(next_token))  # 按照字典id找到预测的下一个字
                count += 1
            pass
        else:  # 不用韵
            while not (self.base_filter(word)) and count < 10000:  # 若生成的字符不满足基础要求 则重新生成
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)  # 从备选里面sample一个
                word = self.tokenizer.convert_ids_to_tokens(int(next_token))  # 按照字典id找到预测的下一个字
                count += 1


        if count >= 1000:  # 实在sample不到
            print(f"{self.foot}:{count},{word}")
            return -1

        if word != '，' and word != '。':  # 标点符号不算
            if word in self.word_dict:  # 记录诗句的用字情况
                self.word_dict[word] += 1
            else:
                self.word_dict[word] = 1

            self.last_word = word  # 更新last_word

        self.idx += 1
        return next_token

    '''选出top_k个备选字'''

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    '''句子sample选用'''

    def sample_sequence(self, model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu'):
        context = torch.tensor(context, dtype=torch.long, device=device)  # 转换成Tensor类型
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        with torch.no_grad():
            for _ in trange(length):

                inputs = {'input_ids': generated}
                outputs = model(**inputs)
                next_token_logits = outputs[0][0, -1, :] / temperature  # 预测下一个字的概率分布
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)  # 挑top_k个作为备选
                next_token = self.sample(filtered_logits)  # sample选出下一个字

                if next_token == -1:  # 如果实在找不到韵脚
                    count = 0
                    ping_dict = [word[0] for word in self.rhyme2word[self.foot][0]]
                    str = shuffle_str(ping_dict)  # 打乱一下字符 免得每次按顺序找到的是同一个字
                    for word in str:  # 直接跳过模型 从韵表里面取
                        count += 1

                        if self.base_filter(word) and self.rhyme_filter(word) and \
                                self.tokenizer._convert_token_to_id(word) != 100:  # 这个韵脚不能是生僻字 最好在训练数据里面有
                            self.idx += 1  # 找到了 idx++
                            next_token = self.tokenizer._convert_token_to_id(word)
                            next_token = torch.tensor([next_token], dtype=torch.long, device=device)
                            print(f"韵脚找了：{count}次，{word}，{next_token}")
                            break

                    if count == len(str):
                        print(self.idx)
                        print(self.tune[self.idx])

                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)  # 更新已经生成的诗文

        return generated.tolist()

    '''根据给出的第一句，生成诗句'''

    def predict_sen(self, text):
        '''此例中，即根据给出的第一句诗句（含逗号），来生成古诗'''
        max_len = self.yan + 1
        if len(text) != max_len:
            print('长度不符')
            return

        os.environ["CUDA_VISIBLE_DEVICES"] = self.device  # 此处设置程序使用哪些显卡

        temperature = self.temperature
        topk = self.topk
        topp = self.topp

        self.init_step(text, max_len)  # 初始化
        self.last_word = text[-2]

        if self.ru == 0 and self.use_rhyme != 0:  # 首句入韵
            print(text)
            self.foot = get_word_foots(text[-2], self.word2rhyme)[0]
            print("韵脚：", self.foot)
            self.jiao.add(text[-2])

        device = "cuda" if torch.cuda.is_available() else "cpu"  # 是否用gpu

        model = GPT2LMHeadModel.from_pretrained(self.model_path)  # 加载模型
        model.to(device)
        model.eval()

        context_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))  # 给出的第一句text：token转换成id数字

        # 预测其余字
        out = self.sample_sequence(
            model=model, length=(1 + self.jue) * 4 * max_len - max_len,
            context=context_tokens,
            temperature=temperature, top_k=topk, top_p=topp, device=device
        )

        ans_text = self.tokenizer.convert_ids_to_tokens(out[0])  # id转成字典里面的token

        ans_text = ''.join(ans_text).replace('##', '').strip()  # 去掉所有的占位符'##'

        return self.use_rhyme, self.foot, ans_text

    '''根据给出的首个文字，生成诗句'''

    def predict_first(self, word):

        if len(word) != 1:
            print('长度不符')
            return
        max_len = self.yan + 1

        os.environ["CUDA_VISIBLE_DEVICES"] = self.device  # 此处设置程序使用哪些显卡

        temperature = self.temperature
        topk = self.topk
        topp = self.topp

        self.init_step(word, 1)  # 初始化
        self.last_word = word

        device = "cuda" if torch.cuda.is_available() else "cpu"  # 是否用gpu

        model = GPT2LMHeadModel.from_pretrained(self.model_path)  # 加载模型
        model.to(device)
        model.eval()

        context_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))  # 给出的第一句text：token转换成id数字

        # 预测其余字
        out = self.sample_sequence(
            model=model, length=(self.jue + 1) * 4 * max_len - 1,
            context=context_tokens,
            temperature=temperature, top_k=topk, top_p=topp, device=device
        )

        ans_text = self.tokenizer.convert_ids_to_tokens(out[0])  # id转成字典里面的token

        ans_text = ''.join(ans_text).replace('##', '').strip()  # 去掉所有的占位符'##'

        return self.use_rhyme, self.foot, ans_text

    '''根据给4个字，生成藏头诗绝句'''

    def predict_hide(self, text):

        if len(text) != 4:
            print('藏头诗的输入必须是4个字！')
            return

        self.jue = 0  # 藏头诗只生成绝句
        max_len = self.yan + 1

        os.environ["CUDA_VISIBLE_DEVICES"] = self.device  # 此处设置程序使用哪些显卡

        temperature = self.temperature
        topk = self.topk
        topp = self.topp

        self.init_step(text, 1)  # 初始化
        self.last_word = text[0]

        device = "cuda" if torch.cuda.is_available() else "cpu"  # 是否用gpu

        model = GPT2LMHeadModel.from_pretrained(self.model_path)  # 加载模型
        model.to(device)
        model.eval()

        context_tokens = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(text[0]))  # 给出的第一个字text[0]：token转换成id数字

        next_text = self.sample_sequence(  # 完成第一句
            model=model, length=max_len - 1,
            context=context_tokens,
            temperature=temperature, top_k=topk, top_p=topp, device=device
        )
        context_tokens = next_text[0]

        for i in range(3):
            context_tokens += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text[i + 1]))  # 第i+1句的第一个字
            self.idx += 1

            next_text = self.sample_sequence(  # 完成第i+1句
                model=model, length=max_len - 1,
                context=context_tokens,
                temperature=temperature, top_k=topk, top_p=topp, device=device
            )
            context_tokens = next_text[0]

        ans_text = self.tokenizer.convert_ids_to_tokens(context_tokens)  # id转成字典里面的token

        ans_text = ''.join(ans_text).replace('##', '').strip()  # 去掉所有的占位符'##'

        return self.use_rhyme, self.foot, ans_text