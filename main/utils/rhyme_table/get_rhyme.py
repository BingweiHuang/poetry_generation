import json
from poetry_generation.settings import BASE_DIR

with open(f"{BASE_DIR}/main/utils/rhyme_table/pingshui/word2ps_rhyme.json", "r", encoding="utf-8") as f:
    word2pingshui = json.load(f)

with open(f"{BASE_DIR}/main/utils/rhyme_table/pingshui/ps_rhyme2word.json", "r", encoding="utf-8") as f:
    pingshui2word = json.load(f)

with open(f"{BASE_DIR}/main/utils/rhyme_table/xinyun/word2xin_rhyme.json", "r", encoding="utf-8") as f:
    word2xinyun = json.load(f)

with open(f"{BASE_DIR}/main/utils/rhyme_table/xinyun/xin_rhyme2word.json", "r", encoding="utf-8") as f:
    xinyun2word = json.load(f)

'''根据字获取韵'''
def get_rhyme(word, word2rhyme):
    if word in word2rhyme:
        return word2rhyme[word]
    return []

'''返回字的平仄情况'''
def get_word_rhyme(word, word2rhyme):
    diaos = get_rhyme(word, word2rhyme)
    ping = []
    ze = []
    for diao in diaos:
        if diao[0] == '平声':
            ping.append(diao[1])
        else:
            ze.append(diao[1])
    return ping, ze

'''返回字的平韵'''
def get_word_foots(word, word2rhyme):
    diaos = get_rhyme(word, word2rhyme)
    foots = []
    for diao in diaos:
        if diao[0] == '平声':
            foots.append(diao[1])

    return foots

# 格律诗平仄音调表
flat_oblique_tone_table = [
    [  # 七言
        [  # 七绝
            [  # 首句押韵
                'x100x10/x0x1100/x0x1x01/x100x10/',  # 仄起
                'x0x1100/x100x10/x1x0011/x0x1100/',  # 平起
            ],
            [  # 首句不押韵
                'x1x0011/x0x1100/x0x1x01/x100x10/',  # 仄起
                'x0x1x01/x100x10/x1x0011/x0x1100/',  # 平起
            ]
        ],
        [  # 七律
            [  # 首句押韵
                'x100x10/x0x1100/x0x1x01/x100x10/x1x0011/x0x1100/x0x1x01/x100x10/',  # 仄起
                'x0x1100/x100x10/x1x0011/x0x1100/x0x1x01/x100x10/x1x0011/x0x1100/',  # 平起
            ],
            [  # 首句不押韵
                'x1x0011/x0x1100/x0x1x01/x100x10/x1x0011/x0x1100/x0x1x01/x100x10/',  # 仄起
                'x0x1x01/x100x10/x1x0011/x0x1100/x0x1x01/x100x10/x1x0011/x0x1100/',  # 平起
            ]
        ]
    ],
    [  # 五言
        [  # 五绝
            [  # 首句押韵
                'x1100/00x10/x0011/x1100/',  # 仄起
                '00x10/x1100/x1x01/00x10/',  # 平起
            ],
            [  # 首句不押韵
                'x1x01/00x10/x0011/x1100/',  # 仄起
                'x0011/x1100/x1x01/00x10/',  # 平起
            ]
        ],
        [  # 五律
            [  # 首句押韵
                'x1100/00x10/x0011/x1100/x1x01/00x10/x0011/x1100/',  # 仄起
                '00x10/x1100/x1x01/00x10/x0011/x1100/x1x01/00x10/',  # 平起
            ],
            [  # 首句不押韵
                'x1x01/00x10/x0011/x1100/x1x01/00x10/x0011/x1100/',  # 仄起
                'x0011/x1100/x1x01/00x10/x0011/x1100/x1x01/00x10/',  # 平起
            ]
        ],
    ]
]

# 格律诗首句平仄音调表
first_sentence_tone_table = [
  [ # 五言
      [ # 仄起
          'x1x01', # 不入韵
          'x1100', # 入韵
      ],
      [ # 平起
          'x0011', # 不入韵
          '00x10', # 入韵
      ],
  ],
  [ # 七言
      [ # 仄起
          'x1x0011', # 不入韵
          'x100x10', # 入韵
      ],
      [ # 平起
          'x0x1x01', # 不入韵
          'x0x1100', # 入韵
      ],
  ],
]

def test():
    pass

if __name__ == "__main__":
    pass