import re

from main.utils.rhyme_table.get_rhyme import pingshui2word, word2pingshui, xinyun2word, word2xinyun
from main.utils.rhyme_table.get_rhyme import get_word_rhyme, get_word_foots, flat_oblique_tone_table

def reserved_chinese_word(text): # 只保留汉字
    return re.sub('[^\u4e00-\u9fa5]+', '', text)

def detect_word(tone, word, word2rhyme): # 检测这个字是否符合平仄
    ans = True
    ping, ze = get_word_rhyme(word, word2rhyme)
    if tone == '1' and len(ze) == 0:
        ans = False
    elif tone == '0' and len(ping) == 0:
        ans = False

    return ans, ping, ze

def get_yun_pos(yan, jue, ru): #  获得韵脚位置的下标
    pos_list = [yan * 2 - 1, yan * 4 - 1]
    if jue == 1: # 律句
        pos_list += [yan * 6 -1, yan * 8 - 1]

    if ru == 0: # 首句入韵
        pos_list.append(yan - 1)

    return pos_list


def metric_poetry_detection(yan, jue, ru, qi, text, use_rhyme):
    
    text = reserved_chinese_word(text) # 清洗 只留汉字
    
    if use_rhyme == 1: # 平水韵还是新韵
        word2rhyme = word2pingshui
    elif use_rhyme == 2:
        word2rhyme = word2xinyun

    tone_table = flat_oblique_tone_table[0 if yan == 7 else 1][jue][ru][qi]  # 诗韵表
    tone_table = tone_table.replace('/', '')
    
    yun_pos_list = sorted(get_yun_pos(yan, jue, ru)) # 获取韵脚下标的list
    
    yun_pos_dict = {} # 存韵脚位置的字的所有 平韵
    pz_err_dict = {} # 存平仄错误的位置应为 '平'(0) or '仄'(1)
    duo_yin_pos = [] # 存多音字（平仄音都有）的位置

    foot = get_word_foots(text[yun_pos_list[0]], word2rhyme) # 第一个韵脚

    for idx, word in enumerate(text):
        fuhe, ping, ze = detect_word(tone_table[idx], word, word2rhyme)
        if not fuhe: # 如果平仄不合要求
            if len(ping) > 0:
                pz_err_dict[idx] = 1
            else:
                pz_err_dict[idx] = 0
                
        if idx in yun_pos_list: # 如果idx是韵脚位置
            foots = get_word_foots(word, word2rhyme) # 获取这字的所有 平韵
            yun_pos_dict[idx] = foots
            foot = [f for f in foot if f in foots] # 确保所有韵脚位置的字都有同一个平韵

        if len(ping) > 0 and len(ze) > 0 and tone_table[idx] != 'x':  # 平仄韵都有的多音字
            duo_yin_pos.append(idx)
    
    rhyme_foot = '' # 韵脚
    if len(foot) >0 :
        rhyme_foot = foot[0]
    
    return yun_pos_dict, pz_err_dict, duo_yin_pos, rhyme_foot
    

