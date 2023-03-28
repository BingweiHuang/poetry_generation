from rest_framework.permissions import IsAuthenticated

from main.permissions import IsUserOrReadOnly, StaffOnly, AdminOnly
from main.throttles import UserAIRateThrottle
from main.utils.MyResponse import MyResponse
from main.utils.model.gpt2.gpt2_model import Gpt2Model
from main.utils.model.lstm.lstm_model import LstmModel
from poetry_generation.settings import BASE_DIR
from rest_framework.response import Response
from rest_framework.views import APIView
import traceback

class Gpt2View(APIView):

    # /create/gpt2?style=1&text=戌时皓月照空白&num=7&use_rhyme=2&yan=7&jue=1&ru=0&qi=0

    permission_classes = ([IsAuthenticated, StaffOnly])
    throttle_scope = 'AI_api' # 针对接口限流
    throttle_classes = [UserAIRateThrottle] # 针对用户限流

    def get(self, request):
        arg = request.GET
        try:

            yan = 7  # 7:七言 5:五言
            jue = 0  # 0:绝句 1:律
            ru = 0  # 0:首句入韵 1:首句不入
            qi = 0  # 0:仄起 1:平起
            use_rhyme = 1  # 压哪个韵 0:不压 1:平水 2:新韵

            style = int(arg.get("style", 0)) # 0:按一个字续写 1:按首句续写 2:藏头诗
            text = arg.get("text", "春")
            num = int(arg.get("num", 1)) # 生成几首
            yan = int(arg.get("yan", yan))
            jue = int(arg.get("jue", jue))
            ru = int(arg.get("ru", ru))
            qi = int(arg.get("qi", qi))
            use_rhyme = int(arg.get("use_rhyme", use_rhyme))
            where = f'{yan}_yan_{"jue" if jue == 0 else "lv"}_model'
            model_path = f'{BASE_DIR}/main/utils/model/gpt2/model_file/{where}'  # 模型路径
            raw_data_path = ''  # 原始训练语料

            model = Gpt2Model(
                yan, jue, ru, qi, use_rhyme, model_path, raw_data_path
            )

            createList = []
            for i in range(num):
                if style == 0:  # 给出第一个字进行预测
                    ret_use_rhyme, foot, sen = model.predict_first(text)
                elif style == 1:  # 给出第一句话进行预测
                    # text += '，'
                    ret_use_rhyme, foot, sen = model.predict_sen(text + '，')
                elif style == 2:  # 藏头诗
                    ret_use_rhyme, foot, sen = model.predict_hide(text)

                sen = sen.split('。')[:-1]
                sen = [s + '。' for s in sen]
                createList.append((ret_use_rhyme, foot, sen))

            datas = {
                "createList": createList,
            }
            return MyResponse(datas, status=200)

        except Exception as e:
            traceback.print_exc()
            return MyResponse({'result': "gpt2作诗失败"}, status=500)

class LstmView(APIView):

    # style=0&text=昨&num=5&use_rhyme=1&yan=7&jue=1&ru=1&qi=0
    permission_classes = ([IsAuthenticated, StaffOnly])
    throttle_scope = 'AI_api'  # 针对接口限流
    throttle_classes = [UserAIRateThrottle]  # 针对用户限流

    def get(self, request):
        arg = request.GET
        try:
            yan = 7  # 7:七言 5:五言
            jue = 0  # 0:绝句 1:律
            ru = 0  # 0:首句入韵 1:首句不入
            qi = 0  # 0:仄起 1:平起
            use_rhyme = 1  # 压哪个韵 0:不压 1:平水 2:新韵

            style = int(arg.get("style", 0))  # 0:按一个字续写 1:按首句续写 2:藏头诗
            text = arg.get("text", "春")
            num = int(arg.get("num", 1))  # 生成几首
            yan = int(arg.get("yan", yan))
            jue = int(arg.get("jue", jue))
            ru = int(arg.get("ru", ru))
            qi = int(arg.get("qi", qi))
            use_rhyme = int(arg.get("use_rhyme", use_rhyme))

            which = f'{yan}_yan_{"jue" if jue == 0 else "lv"}'
            model_path = f'{BASE_DIR}/main/utils/model/lstm/model_file/{which}_model.h5'  # 模型路径
            raw_data_path = f'{BASE_DIR}/main/utils/model/lstm/model_file/{which}_data.json'  # 原始训练语料

            model = LstmModel(
                yan, jue, ru, qi, use_rhyme, model_path, raw_data_path
            )

            createList = []

            for i in range(num):
                if style == 0: # 给出第一个字进行预测
                    ret_use_rhyme, foot, sen = model.predict_first(text)
                elif style == 1: # 给出第一句话进行预测
                    # text += '，'
                    ret_use_rhyme, foot, sen = model.predict_sen(text + '，')
                elif style == 2: # 藏头诗
                    ret_use_rhyme, foot, sen = model.predict_hide(text)

                sen = sen.split('。')[:-1]
                sen = [s + '。' for s in sen]
                createList.append((ret_use_rhyme, foot, sen))

            datas = {
                "createList": createList,
            }
            return MyResponse(datas, 200)

        except Exception as e:
            traceback.print_exc()
            return MyResponse({'result': "lstm作诗失败"}, status=500)