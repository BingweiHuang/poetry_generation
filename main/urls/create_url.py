# blog/urls.py

from django.urls import re_path
from rest_framework.urlpatterns import format_suffix_patterns

from main.views.create_views import Gpt2View
from main.views.create_views import LstmView

urlpatterns = [

    re_path(r'^gpt2/$', Gpt2View.as_view()),
    re_path(r'^lstm/$', LstmView.as_view()),

]

urlpatterns = format_suffix_patterns(urlpatterns)


'''
# 5种方法都开放
urlpatterns += router.urls
'''
